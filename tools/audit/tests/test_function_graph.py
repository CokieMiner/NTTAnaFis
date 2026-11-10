"""Function declarations, source resolution, cfg alternatives and graph cycles."""

import tempfile
import unittest
from pathlib import Path

from tools.audit.function_graph import FunctionGraph


class FunctionGraphTests(unittest.TestCase):
    def graph(self, sources):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for relative, text in sources.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return FunctionGraph(root)

    def test_reexports_self_typed_receivers_fields_and_function_values(self):
        graph = self.graph({
            "src/lib.rs": "mod unit; pub use unit::{Public, entry};",
            "src/unit/mod.rs": "mod owner; mod consumer; pub use owner::{Public, helper as alias}; pub use consumer::entry;",
            "src/unit/owner.rs": """
                pub struct Public { inner: Hidden }
                pub struct Hidden;
                impl Public {
                    pub fn run(&self) { self.inner.run(); self.local(); Self::finish(); }
                    fn local(&self) { helper(); }
                    fn finish() {}
                    pub fn inner(&self) {}
                }
                impl Hidden { pub fn run(&self) {} }
                pub fn helper() {}
                fn callback() {}
                const CALLBACK: fn() = callback;
            """,
            "src/unit/consumer.rs": "use super::{Public, alias}; pub fn entry(value: &Public) { value.run(); alias(); }",
            "tests/use_api.rs": "use ntt_anafis::entry; fn main() { entry(); }",
        })
        functions = list(graph.index.functions.values())
        run = next(f for f in functions if f.name == "run" and f.owner.endswith("::Public"))
        calls = {graph.index.functions[f].symbol for f in graph.calls[run.id]}
        self.assertEqual(calls, {"src::unit::owner::Hidden::run", "src::unit::owner::Public::local", "src::unit::owner::Public::finish"})
        self.assertTrue(run.exported)
        field_getter = next(f for f in functions if f.name == "inner")
        self.assertFalse(graph.incoming[field_getter.id])
        entry = next(f for f in functions if f.name == "entry")
        self.assertTrue(entry.exported)
        self.assertTrue(any(r.path == "tests/use_api.rs" and r.confidence == "resolved" for r in graph.incoming[entry.id]))
        helper = next(f for f in functions if f.name == "helper")
        self.assertEqual({r.spelling for r in graph.incoming[helper.id]}, {"alias", "helper"})
        callback = next(f for f in functions if f.name == "callback")
        self.assertEqual([(r.caller, r.kind, r.confidence) for r in graph.incoming[callback.id]], [(None, "reference", "resolved")])

    def test_all_cfg_variants_macros_turbofish_inline_modules_and_decoys(self):
        graph = self.graph({"src/lib.rs": "mod unit;", "src/unit.rs": """
            // fn fake_comment() { real(); }
            const TEXT: &str = r#"fn fake_string() { real(); }"#;
            macro_rules! generate { () => { fn fake_macro() { real(); } }; }
            fn real() {}
            #[cfg(first)] fn alternative() {}
            #[cfg(second)] fn alternative() {}
            fn driver() { real::<u8>(); alternative(); generate!(real); }
            mod inner { fn child() { super::real(); } }
        """})
        functions = list(graph.index.functions.values())
        self.assertEqual([f.name for f in functions], ["real", "alternative", "alternative", "driver", "child"])
        driver = next(f for f in functions if f.name == "driver")
        real = next(f for f in functions if f.name == "real")
        self.assertIn(real.id, graph.calls[driver.id])
        alternatives = [f for f in functions if f.name == "alternative"]
        self.assertTrue(all(f.conditional for f in alternatives))
        self.assertTrue(all(any(r.confidence == "possible" for r in graph.incoming[f.id]) for f in alternatives))
        self.assertTrue(any(r.kind == "macro" and r.confidence == "possible" for r in graph.incoming[real.id]))
        child = next(f for f in functions if f.name == "child")
        self.assertEqual(graph.calls[child.id], {real.id})

    def test_trait_and_receiver_uncertainty_do_not_become_resolved_calls(self):
        graph = self.graph({"src/lib.rs": "mod unit;", "src/unit.rs": """
            pub struct Value;
            pub trait Action { fn act(&self); fn default(&self) { self.act(); } }
            impl Action for Value { fn act(&self) {} }
            impl Value { pub fn act(&self) {} }
            fn act() {}
            fn qualified() { <Value as Action>::act(&Value); }
            fn unknown(value: &Value) { let value = external(); value.act(); }
            fn explicit(value: &Value) { value.act(); }
        """})
        functions = list(graph.index.functions.values())
        unknown = next(f for f in functions if f.name == "unknown")
        explicit = next(f for f in functions if f.name == "explicit")
        qualified = next(f for f in functions if f.name == "qualified")
        self.assertFalse(graph.calls[unknown.id])
        self.assertEqual(len(graph.calls[explicit.id]), 0)
        self.assertFalse(graph.calls[qualified.id])
        # A trait implementation and inherent method with the same name remain
        # uncertain until receiver/trait dispatch is resolved by the compiler.
        self.assertTrue(any(u.caller == unknown.id and u.kind == "method" for u in graph.unresolved))

    def test_closure_parameters_destructuring_and_pattern_shadowing_preserve_uncertainty(self):
        graph = self.graph({"src/lib.rs": "mod unit;", "src/unit.rs": """
            pub struct Value;
            impl Value { fn act(&self) {} }
            fn helper() {}
            fn closure(helper: impl Fn()) { helper(); }
            fn nested_closure() { apply(|helper| helper()); }
            fn destructured(value: &Value) { let (value, other) = external(); value.act(); }
            fn iterated(value: &Value) { for value in external() { value.act(); } }
            fn matched(value: &Value) { match external() { Some(value) => value.act(), None => {} } }
        """})
        for function in graph.index.functions.values():
            if function.name in {"closure", "nested_closure", "destructured", "iterated", "matched"}:
                self.assertFalse(graph.calls[function.id], function.name)
                self.assertTrue(any(u.caller == function.id for u in graph.unresolved), function.name)

    def test_cfg_modules_test_modules_and_lint_only_cfg_attributes_have_distinct_roles(self):
        graph = self.graph({
            "src/lib.rs": "#[cfg(test)] mod fixture; #[cfg(feature = \"x\")] mod feature; mod normal;",
            "src/fixture.rs": "fn check() {}",
            "src/feature.rs": "fn gated() {}",
            "src/normal.rs": '#[cfg_attr(lint, allow(example, reason = "fixture"))] fn unchanged() {}',
        })
        functions = {f.name: f for f in graph.index.functions.values()}
        self.assertTrue(functions["check"].test)
        self.assertTrue(functions["gated"].conditional)
        self.assertFalse(functions["unchanged"].conditional)

    def test_recursive_components_and_long_chains_do_not_depend_on_python_recursion(self):
        functions = " ".join(f"fn f{i}() {{ f{i + 1}(); }}" for i in range(1100)) + " fn f1100() {} fn a() { b(); } fn b() { a(); }"
        graph = self.graph({"src/lib.rs": functions})
        by_name = {f.name: f for f in graph.index.functions.values()}
        self.assertEqual(by_name["a"].cycle, by_name["b"].cycle)
        self.assertNotEqual(by_name["f0"].cycle, by_name["f1100"].cycle)
        self.assertEqual(graph.summary()["resolved_calls"], 1102)


if __name__ == "__main__":
    unittest.main()
