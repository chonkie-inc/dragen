//! Conversion utilities between PyValue and JSON.

use littrs::PyValue;

/// Format a PyValue for display.
pub fn format_pyvalue(value: &PyValue) -> String {
    match value {
        PyValue::None => "None".to_string(),
        PyValue::Bool(b) => b.to_string(),
        PyValue::Int(i) => i.to_string(),
        PyValue::Float(f) => f.to_string(),
        PyValue::Str(s) => format!("\"{}\"", s),
        PyValue::List(items) => {
            let formatted: Vec<String> = items.iter().map(format_pyvalue).collect();
            format!("[{}]", formatted.join(", "))
        }
        PyValue::Dict(pairs) => {
            let formatted: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("\"{}\": {}", k, format_pyvalue(v)))
                .collect();
            format!("{{{}}}", formatted.join(", "))
        }
    }
}

/// Convert a PyValue to a plain string (without quotes for strings).
/// Used by finish() to get the final answer as a string.
pub fn pyvalue_to_string(value: &PyValue) -> String {
    match value {
        PyValue::None => "None".to_string(),
        PyValue::Bool(b) => b.to_string(),
        PyValue::Int(i) => i.to_string(),
        PyValue::Float(f) => f.to_string(),
        PyValue::Str(s) => s.clone(),
        PyValue::List(items) => {
            let formatted: Vec<String> = items.iter().map(pyvalue_to_string).collect();
            format!("[{}]", formatted.join(", "))
        }
        PyValue::Dict(pairs) => {
            let formatted: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, pyvalue_to_string(v)))
                .collect();
            formatted.join("\n")
        }
    }
}

/// Convert a PyValue to a serde_json::Value for typed deserialization.
pub fn pyvalue_to_json(value: &PyValue) -> serde_json::Value {
    match value {
        PyValue::None => serde_json::Value::Null,
        PyValue::Bool(b) => serde_json::Value::Bool(*b),
        PyValue::Int(i) => serde_json::Value::Number((*i).into()),
        PyValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        PyValue::Str(s) => serde_json::Value::String(s.clone()),
        PyValue::List(items) => {
            serde_json::Value::Array(items.iter().map(pyvalue_to_json).collect())
        }
        PyValue::Dict(pairs) => {
            let map: serde_json::Map<String, serde_json::Value> = pairs
                .iter()
                .map(|(k, v)| (k.clone(), pyvalue_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

/// Convert a serde_json::Value to a PyValue.
pub fn json_to_pyvalue(value: &serde_json::Value) -> PyValue {
    match value {
        serde_json::Value::Null => PyValue::None,
        serde_json::Value::Bool(b) => PyValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PyValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                PyValue::Float(f)
            } else {
                PyValue::None
            }
        }
        serde_json::Value::String(s) => PyValue::Str(s.clone()),
        serde_json::Value::Array(arr) => PyValue::List(arr.iter().map(json_to_pyvalue).collect()),
        serde_json::Value::Object(map) => PyValue::Dict(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_pyvalue(v)))
                .collect(),
        ),
    }
}
