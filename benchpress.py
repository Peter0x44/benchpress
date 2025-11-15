#!/usr/bin/env python3
"""benchpress - Compiler Benchmark Harness Generator"""
import argparse
import re
import sys
import tempfile
import os
from dataclasses import dataclass
from typing import List, Tuple
from pycparser import parse_file, c_generator, c_ast

@dataclass
class Config:
    label: str
    compiler: str
    flags: str

class FunctionExtractor(c_ast.NodeVisitor):
    def __init__(self):
        self.functions = {}
    
    def visit_FuncDef(self, node):
        func_name = node.decl.name
        generator = c_generator.CGenerator()
        
        # Get return type
        return_type = self._get_return_type(node.decl.type)
        # Get params  
        params = self._get_params(node.decl.type, generator)
        # Get body
        body = generator.visit(node.body)
        
        self.functions[func_name] = {
            'return_type': return_type,
            'params': params,
            'body': body
        }
    
    def _get_return_type(self, func_type):
        if hasattr(func_type, 'type'):
            if hasattr(func_type.type, 'names'):
                return ' '.join(func_type.type.names)
            elif hasattr(func_type.type, 'type') and hasattr(func_type.type.type, 'names'):
                return ' '.join(func_type.type.type.names)
        return 'void'
    
    def _get_params(self, func_type, generator):
        if hasattr(func_type, 'args') and func_type.args and hasattr(func_type.args, 'params'):
            return ', '.join(generator.visit(p) for p in func_type.args.params)
        return 'void'

class MarkerExtractor:
    def __init__(self, source: str):
        self.source = source
    
    def extract_all(self):
        # Find function names
        benchfunc_name = self._find_name('BENCHFUNC')
        warmup_name = self._find_name('WARMUP')
        benchmark_name = self._find_name('BENCHMARK')
        
        # Remove markers for parsing
        parseable = re.sub(r'\bBENCHFUNC\s+', '', self.source)
        parseable = re.sub(r'\bWARMUP\s+', '', parseable)
        parseable = re.sub(r'\bBENCHMARK\s+', '', parseable)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            temp_path = f.name
            f.write(parseable)
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fake_libc = os.path.join(script_dir, 'fake_libc_include')
            
            ast = parse_file(temp_path, use_cpp=True, cpp_path='gcc',
                           cpp_args=['-E', f'-I{fake_libc}', '-D__attribute__(x)=',
                                   '-D__restrict=restrict', '-D__extension__='])
            
            extractor = FunctionExtractor()
            extractor.visit(ast)
            
            benchfunc = extractor.functions[benchfunc_name]
            warmup = extractor.functions[warmup_name]
            benchmark = extractor.functions[benchmark_name]
            
            benchfunc_info = (benchfunc['return_type'], benchfunc_name, 
                            benchfunc['params'], benchfunc['body'])
            warmup_info = (warmup_name, warmup['body'])
            benchmark_info = (benchmark_name, benchmark['body'])
        finally:
            os.unlink(temp_path)
        
        helpers = self._extract_helpers()
        return benchfunc_info, warmup_info, benchmark_info, helpers
    
    def _find_name(self, marker):
        m = re.search(rf'{marker}\s+[\w\s\*]+?\s+(\w+)\s*\(', self.source)
        if not m:
            raise ValueError(f"Could not find {marker} marker")
        return m.group(1)
    
    def _extract_helpers(self):
        helpers = self.source
        for marker in ['BENCHFUNC', 'WARMUP', 'BENCHMARK']:
            pattern = rf'{marker}\s+[^{{]+\{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}}'
            helpers = re.sub(pattern, '', helpers, flags=re.DOTALL)
        return re.sub(r'\n\n\n+', '\n\n', helpers).strip()

class TemplateParser:
    def __init__(self, source: str):
        self.extractor = MarkerExtractor(source)
    
    def parse_all(self):
        return self.extractor.extract_all()

class CodeGenerator:
    def __init__(self, input_filename: str, compare_mode: str = None):
        self.input_filename = input_filename
        self.compare_mode = compare_mode
    
    def generate(self, configs, benchfunc_info, warmup_info, benchmark_info, helpers):
        return_type, func_name, params, func_body = benchfunc_info
        warmup_name, warmup_body = warmup_info
        benchmark_name, benchmark_body = benchmark_info
        
        parts = [
            self._gen_header(configs),
            "\n// ========== BENCHFUNC Implementations ==========\n",
            self._gen_benchfunc(configs, return_type, func_name, params, func_body),
            "\n// ========== Test Harness ==========\n",
            self._gen_harness(configs, return_type, func_name, params,
                            warmup_name, warmup_body, benchmark_name, benchmark_body,
                            helpers)
        ]
        return '\n'.join(parts)
    
    def _gen_header(self, configs):
        lines = ["#if 0", "#!/bin/sh", "# Self-building benchmark harness",
                f"# Generated by benchpress from: {self.input_filename}", "",
                "CC_GCC=${CC_GCC:-gcc}", "CC_CLANG=${CC_CLANG:-clang}", "",
                "# Compile BENCHFUNC with each configuration"]
        
        for c in configs:
            compiler_var = f"$CC_{c.compiler.upper()}"
            lines.append(f'{compiler_var} {c.flags} -DCOMPILE_{c.label} -c -o func_{c.label}.o $0 || exit 1')
        
        obj_files = ' '.join(f'func_{c.label}.o' for c in configs)
        lines.extend(["", "# Compile and link test harness",
                     f'$CC_GCC -DTEST_HARNESS -O2 -o benchmark $0 {obj_files} -lm || exit 1',
                     "", './benchmark', "", 'exit 0', '#endif'])
        return '\n'.join(lines)
    
    def _gen_benchfunc(self, configs, return_type, func_name, params, body):
        body = body.strip()
        if body.startswith('{') and body.endswith('}'):
            body = body[1:-1].strip()
        
        lines = []
        for c in configs:
            lines.extend([f"#ifdef COMPILE_{c.label}",
                        f"{return_type} {func_name}_{c.label}({params}) {{",
                        self._indent(body), "}",
                        f"#endif // COMPILE_{c.label}", ""])
        return '\n'.join(lines)
    
    def _gen_harness(self, configs, return_type, func_name, params,
                    warmup_name, warmup_body, benchmark_name, benchmark_body, helpers):
        lines = [
            "#ifdef TEST_HARNESS",
            "",
            "#include <stdio.h>",
            "#include <stdint.h>",
            "#include <time.h>",
            "#include <string.h>",
            "",
            "// Define markers as empty",
            "#define BENCHFUNC",
            "#define WARMUP",
            "#define BENCHMARK",
            "",
            "// Timing function",
            "int64_t get_nanos(void) {",
            "    struct timespec ts;",
            "    clock_gettime(CLOCK_MONOTONIC, &ts);",
            "    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;",
            "}",
            "",
            "// Helper code",
            helpers,
            "",
            "// Forward declarations"
        ]
        for c in configs:
            lines.append(f"{return_type} {func_name}_{c.label}({params});")
        
        lines.extend(["", "// Generate WARMUP/BENCHMARK wrappers for each config", ""])
        
        warmup_body = self._strip_braces(warmup_body)
        benchmark_body = self._strip_braces(benchmark_body)
        
        for c in configs:
            lines.extend([
                f"// Wrappers for {c.label}",
                f"#define {func_name} {func_name}_{c.label}",
                f"void {warmup_name}_{c.label}(void) {{",
                self._indent(warmup_body), "}",
                f"void {benchmark_name}_{c.label}(void) {{",
                self._indent(benchmark_body), "}",
                f"#undef {func_name}", ""
            ])
        
        lines.extend([
            "typedef struct {",
            "    const char *label;",
            "    void (*warmup)(void);",
            "    void (*benchmark)(void);",
            "} TestConfig;", "",
            "int main(void) {",
            "    TestConfig configs[] = {"
        ])
        
        for c in configs:
            lines.append(f'        {{"{c.compiler} {c.flags}", {warmup_name}_{c.label}, {benchmark_name}_{c.label}}},')
        
        lines.extend([
            "    };",
            f"    int num_configs = {len(configs)};",
            "    int64_t times[num_configs];", "",
            "    // Run all benchmarks and print results immediately",
            "    for (int i = 0; i < num_configs; i++) {",
            "        configs[i].warmup();",
            "        int64_t start = get_nanos();",
            "        configs[i].benchmark();",
            "        times[i] = get_nanos() - start;",
            '        printf("%s: %.3f seconds\\n", configs[i].label, times[i] / 1e9);',
            "    }", "",
        ])
        
        # Generate comparison code based on mode
        if self.compare_mode == 'across':
            # Compare same flags across different compilers
            # Group configs by flags
            lines.extend([
                "    // Compare same flags across compilers",
                "    // Group configs by flags",
            ])
            
            # Extract unique flag sets from configs
            flags_map = {}
            for i, c in enumerate(configs):
                if c.flags not in flags_map:
                    flags_map[c.flags] = []
                flags_map[c.flags].append(i)
            
            # Generate comparison code for each flag set
            for flags, indices in flags_map.items():
                if len(indices) > 1:
                    lines.extend([
                        f"    // Compare configs with flags: {flags}",
                        "    {",
                        f"        int flag_configs[] = {{{', '.join(str(i) for i in indices)}}};",
                        f"        int count = {len(indices)};",
                        "        int fastest_idx = flag_configs[0];",
                        "        for (int i = 1; i < count; i++) {",
                        "            if (times[flag_configs[i]] < times[fastest_idx]) {",
                        "                fastest_idx = flag_configs[i];",
                        "            }",
                        "        }",
                        "",
                        "        double speedup = 1.0;",
                        "        for (int i = 0; i < count; i++) {",
                        "            int idx = flag_configs[i];",
                        "            if (idx != fastest_idx) {",
                        "                double ratio = (double)times[idx] / times[fastest_idx];",
                        "                if (ratio > speedup) speedup = ratio;",
                        "            }",
                        "        }",
                        f'        printf("{flags}: %s was %.2fx faster\\n", configs[fastest_idx].label, speedup);',
                        "    }",
                    ])
        elif self.compare_mode:
            # Specific label comparisons
            compare_labels = [label.strip() for label in self.compare_mode.split(',')]
            lines.extend([
                "    // Compare specific configs",
                "    const char *compare_labels[] = {",
            ])
            for label in compare_labels:
                lines.append(f'        "{label}",')
            lines.extend([
                "    };",
                f"    int num_compare = {len(compare_labels)};",
                "    int compare_indices[num_compare];",
                "    int found = 0;",
                "",
                "    // Find indices of specified labels",
                "    for (int i = 0; i < num_compare; i++) {",
                "        compare_indices[i] = -1;",
                "        for (int j = 0; j < num_configs; j++) {",
                "            if (strcmp(configs[j].label, compare_labels[i]) == 0) {",
                "                compare_indices[i] = j;",
                "                found++;",
                "                break;",
                "            }",
                "        }",
                "    }",
                "",
                "    if (found > 1) {",
                "        // Find fastest among specified configs",
                "        int fastest_compare = -1;",
                "        for (int i = 0; i < num_compare; i++) {",
                "            if (compare_indices[i] >= 0) {",
                "                if (fastest_compare < 0 || times[compare_indices[i]] < times[compare_indices[fastest_compare]]) {",
                "                    fastest_compare = i;",
                "                }",
                "            }",
                "        }",
                "",
                '        printf("Specific comparisons:\\n");',
                "        for (int i = 0; i < num_compare; i++) {",
                "            if (compare_indices[i] >= 0 && i != fastest_compare) {",
                "                int idx = compare_indices[i];",
                "                int fastest_idx = compare_indices[fastest_compare];",
                "                double speedup = (double)times[idx] / times[fastest_idx];",
                '                printf("  %s is %.2fx faster than %s (%.1f%% faster)\\n",',
                "                       configs[fastest_idx].label, speedup, configs[idx].label,",
                "                       (speedup - 1.0) * 100.0);",
                "            }",
                "        }",
                '        printf("\\n");',
                "    }",
            ])
        
        lines.extend([
            "",
            "    return 0;",
            "}", "",
            "#endif // TEST_HARNESS"
        ])
        return '\n'.join(lines)
    
    def _strip_braces(self, code):
        code = code.strip()
        if code.startswith('{') and code.endswith('}'):
            return code[1:-1].strip()
        return code
    
    def _indent(self, code, spaces=4):
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in code.split('\n'))

def parse_config_spec(spec):
    parts = spec.split(':', 2)
    if len(parts) == 2:
        compiler, flags = parts
        label = f"{compiler}_{flags.replace('-', '').replace(' ', '_')}"
    elif len(parts) == 3:
        label, compiler, flags = parts
    else:
        raise ValueError(f"Invalid config spec: {spec}")
    
    if compiler not in ['gcc', 'clang']:
        raise ValueError(f"Invalid compiler: {compiler}")
    return Config(label=label, compiler=compiler, flags=flags)

def get_default_configs(no_gcc, no_clang):
    configs = []
    if not no_gcc:
        configs.append(Config(label="gcc_O3", compiler="gcc", flags="-O3 -march=native"))
    if not no_clang:
        configs.append(Config(label="clang_O3", compiler="clang", flags="-O3 -march=native"))
    if not configs:
        raise ValueError("At least one compiler must be enabled")
    return configs

def main():
    parser = argparse.ArgumentParser(
        description='Generate self-building compiler benchmark harness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
examples:
  # Compare gcc and clang with -O2 and -O3
  benchpress template.c --compilers gcc:clang --flags "-O2:-O3" -o bench.c
  
  # Specific configs with custom labels
  benchpress template.c --config "fast:gcc:-O3 -march=native" --config clang:-O2 -o bench.c
  
  # Custom comparisons
  benchpress template.c --compilers gcc:clang --flags "-O2:-O3" --compare "gcc -O3,clang -O3" -o bench.c
        '''
    )
    parser.add_argument('input', help='template file with BENCHFUNC/WARMUP/BENCHMARK markers')
    parser.add_argument('-o', '--output', required=True, help='output self-building benchmark file')
    parser.add_argument('--config', action='append', dest='configs', metavar='SPEC',
                       help='add config: [label:]compiler:flags (can be repeated)')
    parser.add_argument('--compilers', metavar='LIST', 
                       help='compilers to test, colon-separated (e.g., gcc:clang)')
    parser.add_argument('--flags', metavar='LIST',
                       help='flag sets to test, colon-separated (e.g., "-O2:-O3")')
    parser.add_argument('--compare', metavar='LABELS',
                       help='specific configs to compare, comma-separated (e.g., "gcc -O3,clang -O3")')
    
    args = parser.parse_args()
    
    # Handle different config modes
    if args.compilers and args.flags:
        # Generate all combinations of compilers and flags
        compilers = args.compilers.split(':')
        flag_sets = args.flags.split(':')
        configs = []
        for compiler in compilers:
            compiler = compiler.strip()
            if compiler not in ['gcc', 'clang']:
                print(f"Error: Invalid compiler '{compiler}'. Must be 'gcc' or 'clang'", file=sys.stderr)
                return 1
            for flags in flag_sets:
                flags = flags.strip()
                label = f"{compiler}_{flags.replace('-', '').replace(' ', '_')}"
                configs.append(Config(label=label, compiler=compiler, flags=flags))
    elif args.configs:
        configs = [parse_config_spec(s) for s in args.configs]
    else:
        # Default: gcc and clang with -O3
        configs = get_default_configs(False, False)
    
    try:
        with open(args.input) as f:
            template_source = f.read()
    except IOError as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        return 1
    
    try:
        template_parser = TemplateParser(template_source)
        benchfunc_info, warmup_info, benchmark_info, helpers = template_parser.parse_all()
    except Exception as e:
        print(f"Error parsing template: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Determine comparison mode
    if args.compare:
        # User provided specific labels
        compare_mode = args.compare
    elif args.compilers and args.flags:
        # Using flag combinations: compare same flags across compilers
        compare_mode = 'across'
    else:
        # Default or individual configs: no comparisons
        compare_mode = None
    
    generator = CodeGenerator(args.input, compare_mode)
    output = generator.generate(configs, benchfunc_info, warmup_info, benchmark_info, helpers)
    
    try:
        with open(args.output, 'w') as f:
            f.write(output)
    except IOError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1
    
    print(f"Generated {args.output}")
    print(f"Run with: sh {args.output}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
