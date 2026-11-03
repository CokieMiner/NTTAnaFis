"""Module registry, test separation, cfg and position-preserving scanner controls."""

import unittest

from tools.audit.items import cfg_requires_test, top_level_items
from tools.audit.rust_source import clean_rust_code
from tools.audit.structure_checks import structural_findings


class StructureTests(unittest.TestCase):
    def findings(self, text, registry=True, path=None):
        path = path or ("src/demo/mod.rs" if registry else "src/demo/value.rs")
        return {f.kind for f in structural_findings(text, path, registry=registry, test_file=False)}

    def test_registry_rules_and_decoys(self):
        valid = '//! Registry.\nuse super::Thing;\nmod value;\npub use value::Value;\n#[cfg(test)]\nmod tests;'
        self.assertEqual(self.findings(valid), set())
        decoys = '// mod fake { todo!(); }\n/* nested /* fn f() {} */ comment */\nmod real;\n#[doc = r#"fn fake() {}"#]\npub use real::Thing;'
        self.assertEqual(self.findings(decoys), set())
        for text, kind in (
            ("mod value; const HIDDEN: u8 = 1;", "implementation_in_module_registry"),
            ("pub\nasync\nfn entry() {}", "implementation_in_module_registry"),
            ("mod tests; mod value;", "test_module_without_test_gate"),
            ("#[cfg(test)] mod tests; mod value;", "module_registry_order"),
            ("generate_code!();", "implementation_in_module_registry"),
            ("mod broken {", "unparsed_rust_structure"),
        ):
            with self.subTest(text=text):
                self.assertIn(kind, self.findings(text))

    def test_registry_only_admits_the_test_module(self):
        for item in ("use std::thread;", "pub use tests::Fixture;", "extern crate std;", "mod fixtures;"):
            for cfg in ("test", 'all(test, feature = "std")'):
                with self.subTest(item=item, cfg=cfg):
                    self.assertIn("test_item_in_module_registry", self.findings(f"#[cfg({cfg})] {item}"))
        self.assertEqual(self.findings("#[cfg(test)] mod tests;"), set())
        kinds = self.findings("#[cfg(test)] mod checks { #[test] fn check() {} }", registry=False)
        self.assertIn("inline_test_module", kinds)
        self.assertIn("test_in_production_file", kinds)

    def test_cfg_requirements(self):
        for code, expected in (
            ('cfg(any(test, feature = "x"))', False),
            ('cfg(all(test, feature = "x"))', True),
            ("cfg(not(test))", False), ("cfg(not(not(test)))", True),
            ("cfg(all())", False), ("cfg(any())", True),
        ):
            with self.subTest(code=code):
                self.assertEqual(cfg_requires_test(code), expected)

    def test_scanner_preserves_positions_and_function_modifiers(self):
        source = 'const TEXT: &str = "first\\\nsecond";\nfn f() {}'
        cleaned = clean_rust_code(source)
        self.assertEqual(len(source), len(cleaned))
        self.assertEqual([i for i,c in enumerate(source) if c == '\n'], [i for i,c in enumerate(cleaned) if c == '\n'])
        for prefix in ("pub const", "pub(crate) const", "pub unsafe", 'pub extern "C"', "async"):
            with self.subTest(prefix=prefix):
                item = top_level_items(f"{prefix} fn entry() {{}}")[0]
                self.assertEqual((item.kind, item.name), ("fn", "entry"))

    def test_architecture_selection_stays_in_arch(self):
        path = "src/plan/demo.rs"
        for text in ('#[cfg(target_arch = "x86_64")] fn f() {}',
                     '#[target_feature(enable = "avx2")] unsafe fn f() {}',
                     'fn f() { is_x86_feature_detected!("adx"); }'):
            with self.subTest(text=text):
                self.assertIn("architecture_selection_outside_arch", self.findings(text, False, path))
                self.assertNotIn("architecture_selection_outside_arch", self.findings(text, False, "src/arch/demo.rs"))
        self.assertEqual(self.findings('#[cfg(target_pointer_width = "16")] fn f() {}', False, path), set())


if __name__ == "__main__":
    unittest.main()
