//! Test rich diagnostic error messages from litter.
//!
//! Run with: cargo run --example test_diagnostics

use litter::{PyValue, Sandbox, ToolInfo};

fn main() {
    let mut sandbox = Sandbox::new();

    // Register a tool with type info
    let search_info = ToolInfo::new("search", "Search the web for information")
        .arg_required("query", "str", "The search query")
        .arg_optional("limit", "int", "Number of results (1-10)")
        .returns("list");

    sandbox.register_tool(search_info, |args| {
        let query = args.get(0).and_then(|v| v.as_str()).unwrap_or("");
        let limit = args.get(1).and_then(|v| v.as_int()).unwrap_or(5);
        println!("Would search for: {} (limit: {})", query, limit);
        PyValue::List(vec![])
    });

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  TESTING RICH DIAGNOSTIC ERROR MESSAGES");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Test 1: Correct usage
    println!("Test 1: Correct usage");
    println!("─────────────────────");
    let result = sandbox.execute(r#"search("AI agents", 5)"#);
    match result {
        Ok(_) => println!("✅ Success\n"),
        Err(e) => println!("❌ Error: {}\n", e),
    }

    // Test 2: Wrong type for first argument (passing int instead of str)
    println!("Test 2: Wrong type for query (int instead of str)");
    println!("─────────────────────────────────────────────────");
    let result = sandbox.execute(r#"search(12345, 5)"#);
    match result {
        Ok(_) => println!("✅ Success\n"),
        Err(e) => println!("{}\n", e),
    }

    // Test 3: Wrong type for second argument (passing str instead of int)
    println!("Test 3: Wrong type for limit (str instead of int)");
    println!("─────────────────────────────────────────────────");
    let result = sandbox.execute(r#"search("AI agents", "five")"#);
    match result {
        Ok(_) => println!("✅ Success\n"),
        Err(e) => println!("{}\n", e),
    }

    // Test 4: Unexpected keyword argument
    println!("Test 4: Unexpected keyword argument");
    println!("────────────────────────────────────");
    let result = sandbox.execute(r#"search("AI agents", timeout=30)"#);
    match result {
        Ok(_) => println!("✅ Success\n"),
        Err(e) => println!("{}\n", e),
    }

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  TESTS COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");
}
