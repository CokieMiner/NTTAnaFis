"""Visibility, helper placement, impl ownership and downward dependency order."""

import tempfile
import unittest
from pathlib import Path

from tools.audit.function_graph import FunctionGraph
from tools.audit.function_rules import function_reviews


class FunctionRuleTests(unittest.TestCase):
    def graph(self, sources):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for relative, text in sources.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return FunctionGraph(root)

    def test_general_visibility_placement_impl_and_leaf_matrix(self):
        graph = self.graph({
            "src/lib.rs": "mod area; pub use area::Public;",
            "src/area/mod.rs": "mod owner; mod first; mod second; #[cfg(test)] mod tests; pub use owner::{Public, only_other, non_leaf, shared_leaf, tested};",
            "src/area/owner.rs": """
                pub struct Public { value: u8 }
                impl Public { pub fn public_api(&self) {} }
                pub struct Machine { value: u8 }
                impl Machine {
                    pub fn unrelated(value: u8) -> u8 { value }
                    pub fn new() -> Self { Self { value: 0 } }
                    pub fn instance(&self) { self.value; }
                }
                pub struct Namespace;
                impl Namespace { pub fn routine() {} }
                pub fn local_pub() {}
                fn local_driver() { local_pub(); }
                pub fn only_other() {}
                pub fn non_leaf() { intermediate(); }
                fn intermediate() {}
                pub fn shared_leaf() {}
                pub fn tested() {}
                fn unobserved() {}
                fn callback() {}
                const POINTER: fn() = callback;
            """,
            "src/area/first.rs": "use super::{only_other, non_leaf, shared_leaf}; fn driver() { only_other(); non_leaf(); shared_leaf(); }",
            "src/area/second.rs": "use super::{non_leaf, shared_leaf}; fn driver() { non_leaf(); shared_leaf(); }",
            "src/area/tests.rs": "use super::owner::tested; #[test] fn check() { tested(); }",
        })
        reviews = function_reviews(graph)
        kinds = {}
        for review in reviews:
            name = graph.index.functions[review.function].name
            kinds.setdefault(name, set()).add(review.kind)
        self.assertIn("possibly_unnecessary_pub", kinds["local_pub"])
        self.assertIn("single_consumer_other_file", kinds["only_other"])
        self.assertIn("non_leaf_without_local_users", kinds["non_leaf"])
        self.assertNotIn("non_leaf_without_local_users", kinds.get("shared_leaf", set()))
        self.assertIn("static_helper_without_type_dependency", kinds["unrelated"])
        for name in ("public_api", "new", "instance", "routine", "tested", "callback"):
            self.assertNotIn("possibly_unnecessary_pub", kinds.get(name, set()))
            self.assertNotIn("static_helper_without_type_dependency", kinds.get(name, set()))
        self.assertIn("no_observed_users", kinds["unobserved"])
        self.assertNotIn("no_observed_users", kinds.get("callback", set()))

    def test_ordering_exempts_cycles_public_entries_traits_and_cfg_alternatives(self):
        graph = self.graph({"src/lib.rs": "mod unit; pub use unit::public_api;", "src/unit.rs": """
            fn leaf() {}
            fn driver() { leaf(); }
            pub fn public_api() {}
            fn api_user() { public_api(); }
            fn cycle_a() { cycle_b(); }
            fn cycle_b() { cycle_a(); }
            #[cfg(first)] fn conditional() {}
            fn conditional_user() { conditional(); }
            pub struct State;
            impl Trait for State { fn act() {} }
            fn good_driver() { good_leaf(); }
            fn good_leaf() {}
        """})
        ordered = [r for r in function_reviews(graph) if r.kind == "callee_precedes_caller"]
        self.assertEqual([graph.index.functions[r.function].name for r in ordered], ["leaf"])
        self.assertIn("src::unit::driver", ordered[0].evidence[0])

    def test_macro_and_dynamic_uses_block_absence_based_visibility_claims(self):
        graph = self.graph({"src/lib.rs": "mod unit;", "src/unit.rs": """
            pub fn macro_helper() {}
            pub struct State { value: u8 }
            impl State { pub fn act(&self) {} }
            fn driver() { generate!(macro_helper); unknown.act(); }
            #[cfg(feature = "x")] pub fn gated() {}
            #[unsafe(no_mangle)] pub extern "C" fn exported_symbol() {}
            macro_rules! generate { ($name:ident) => { $name(); }; }
        """})
        reviews = function_reviews(graph)
        for review in reviews:
            name = graph.index.functions[review.function].name
            if name in {"macro_helper", "act", "gated", "exported_symbol"}:
                self.assertNotIn(review.kind, {"no_observed_users", "possibly_unnecessary_pub"})


if __name__ == "__main__":
    unittest.main()
