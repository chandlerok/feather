//! Entity key encoding.
//!
//! One Valkey hash per entity, keyed by
//! `{project}:{entity_name}:{encoded_entity_key}` where the encoded key is
//!
//! ```text
//! {len}:{value}|{len}:{value}
//! ```
//!
//! Each component carries its own byte length, so a `|` inside a value cannot be
//! mistaken for a separator: the decoder reads a length, consumes exactly that
//! many bytes, then expects a separator. That means no escaping rule to get wrong
//! and no restriction on key contents.
//!
//! There is no cluster hash tag, and that is permanent rather than a v1
//! shortcut. One hash is already one cluster slot, so every field of an entity is
//! colocated by construction. A project-level tag would force every entity in the
//! project into a single slot, which is the opposite of what a cluster is for.

use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, Int32Type, Int64Type};

use crate::error::{Error, Result};

/// Cap on a single entity key component, so one pathological key cannot produce
/// an unbounded Valkey key.
pub const MAX_COMPONENT_LEN: usize = 512;

const SEPARATOR: u8 = b'|';
const LENGTH_TERMINATOR: u8 = b':';

/// Encode an entity tuple. Components may contain arbitrary bytes.
///
/// ```
/// use feather_core::encode_entity_key;
/// assert_eq!(encode_entity_key(&[b"123", b"abc"]).unwrap(), b"3:123|3:abc");
/// ```
pub fn encode_entity_key(components: &[&[u8]]) -> Result<Vec<u8>> {
    encode_entity_key_for("", components)
}

/// As [`encode_entity_key`], but names the entity in any error message.
pub fn encode_entity_key_for(entity: &str, components: &[&[u8]]) -> Result<Vec<u8>> {
    if components.is_empty() {
        return Err(Error::EmptyEntityKey {
            entity: entity.to_owned(),
        });
    }

    let total: usize = components.iter().map(|c| c.len() + 4).sum();
    let mut out = Vec::with_capacity(total);

    for (index, component) in components.iter().enumerate() {
        if component.len() > MAX_COMPONENT_LEN {
            return Err(Error::EntityKeyComponentTooLong {
                entity: entity.to_owned(),
                index,
                len: component.len(),
                max: MAX_COMPONENT_LEN,
            });
        }
        if index > 0 {
            out.push(SEPARATOR);
        }
        out.extend_from_slice(component.len().to_string().as_bytes());
        out.push(LENGTH_TERMINATOR);
        out.extend_from_slice(component);
    }

    Ok(out)
}

/// Decode an encoded entity key back into its components.
///
/// ```
/// use feather_core::decode_entity_key;
/// let decoded = decode_entity_key(b"3:123|3:a|b").unwrap();
/// assert_eq!(decoded, vec![b"123".to_vec(), b"a|b".to_vec()]);
/// ```
pub fn decode_entity_key(encoded: &[u8]) -> Result<Vec<Vec<u8>>> {
    let malformed = |reason: &str| Error::MalformedEntityKey {
        reason: reason.to_owned(),
    };

    if encoded.is_empty() {
        return Err(malformed("empty"));
    }

    let mut components = Vec::new();
    let mut cursor = 0usize;

    loop {
        let digits_start = cursor;
        while cursor < encoded.len() && encoded[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor == digits_start {
            return Err(malformed("expected a length"));
        }
        if cursor >= encoded.len() || encoded[cursor] != LENGTH_TERMINATOR {
            return Err(malformed("expected `:` after a length"));
        }

        // Cap the digit count before parsing so a huge literal cannot overflow.
        if cursor - digits_start > 6 {
            return Err(malformed("component length has too many digits"));
        }
        let len: usize = std::str::from_utf8(&encoded[digits_start..cursor])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| malformed("unparseable length"))?;
        if len > MAX_COMPONENT_LEN {
            return Err(malformed("component length over the cap"));
        }

        cursor += 1; // consume `:`
        let end = cursor
            .checked_add(len)
            .ok_or_else(|| malformed("length overflow"))?;
        if end > encoded.len() {
            return Err(malformed("component runs past the end"));
        }
        components.push(encoded[cursor..end].to_vec());
        cursor = end;

        match encoded.get(cursor) {
            None => break,
            Some(&SEPARATOR) => cursor += 1,
            Some(_) => return Err(malformed("expected `|` between components")),
        }
    }

    Ok(components)
}

/// Build the full Valkey hash key for an entity.
///
/// ```
/// use feather_core::{encode_entity_key, entity_hash_key};
/// let encoded = encode_entity_key(&[b"u1"]).unwrap();
/// assert_eq!(entity_hash_key("ads", "user", &encoded), b"ads:user:2:u1");
/// ```
pub fn entity_hash_key(project: &str, entity_name: &str, encoded: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(project.len() + entity_name.len() + encoded.len() + 2);
    out.extend_from_slice(project.as_bytes());
    out.push(b':');
    out.extend_from_slice(entity_name.as_bytes());
    out.push(b':');
    out.extend_from_slice(encoded);
    out
}

/// Split a hash key back into `(project, entity_name, encoded_key)`.
pub fn parse_entity_hash_key(key: &[u8]) -> Result<(&str, &str, &[u8])> {
    fn malformed(reason: &str) -> Error {
        Error::MalformedEntityKey {
            reason: reason.to_owned(),
        }
    }
    fn as_str(bytes: &[u8]) -> Result<&str> {
        std::str::from_utf8(bytes).map_err(|_| malformed("not UTF-8"))
    }

    let first = key
        .iter()
        .position(|&b| b == b':')
        .ok_or_else(|| malformed("no project separator"))?;
    let rest = &key[first + 1..];
    let second = rest
        .iter()
        .position(|&b| b == b':')
        .ok_or_else(|| malformed("no entity separator"))?;

    Ok((
        as_str(&key[..first])?,
        as_str(&rest[..second])?,
        &rest[second + 1..],
    ))
}

/// The bytes of one entity join-key value, as the key encoder carries them.
///
/// An integer key is its decimal form and a string key is its UTF-8 bytes, which is the shape
/// the encoder's own doctests use.
///
/// Two callers share this conversion, and they are the two sides of one key: the write path
/// reads a source's join key column and encodes it, and the serving read path reads the join key
/// column of a caller's entity frame and encodes it. Both go through here so that a value written
/// under a key and later looked up by the same value cannot disagree about what that key is. A
/// serving caller that built its own key bytes would be the way that agreement breaks, which is
/// why the binding encodes from the frame rather than exposing [`encode_entity_key`] upward.
///
/// Args:
///     column: The column holding the join key.
///     row: The row to read.
///
/// Returns:
///     The component's bytes, ready for [`encode_entity_key`].
///
/// Raises:
///     [`Error::NullEntityKey`] if the value is null. A null cannot address an entity, and
///         encoding it as an empty component would name a different one.
///     [`Error::UnsupportedKeyType`] if the column is neither an integer nor a string. The two
///         types are what the write path can read out of a source and the read path out of a
///         frame, and they are exactly the set
///         [`crate::offline::Engine::scan_latest_per_entity`] admits, so a key that reaches a
///         write or a read is one this function carries.
pub fn entity_key_component(column: &dyn Array, row: usize) -> Result<Vec<u8>> {
    if column.is_null(row) {
        return Err(Error::NullEntityKey);
    }
    match column.data_type() {
        DataType::Int32 => Ok(column
            .as_primitive::<Int32Type>()
            .value(row)
            .to_string()
            .into_bytes()),
        DataType::Int64 => Ok(column
            .as_primitive::<Int64Type>()
            .value(row)
            .to_string()
            .into_bytes()),
        DataType::Utf8 => Ok(column.as_string::<i32>().value(row).as_bytes().to_vec()),
        other => Err(Error::UnsupportedKeyType {
            dtype: other.to_string(),
        }),
    }
}

/// Whether a key has the shape of an entity hash, `project:{entity}:{len}:{value}`.
///
/// Garbage collection walks a project's keyspace and must not send `HDEL` to a key that is
/// not a hash, where the server answers `WRONGTYPE` and a refresh fails over a key that has
/// nothing to do with the view being retired. The encoded tail has to decode, which is the
/// `{len}:` suffix that also keeps `{project}:views` from looking like an entity named
/// `views`.
///
/// Args:
///     key: The key as it exists in the store.
///     project: The project the key is expected to belong to.
///
/// Returns:
///     `true` when the key parses as this project's entity hash.
pub fn is_entity_hash_key(key: &[u8], project: &str) -> bool {
    match parse_entity_hash_key(key) {
        Ok((found, _, encoded)) => found == project && decode_entity_key(encoded).is_ok(),
        Err(_) => false,
    }
}

/// The project registry key, `{project}:views`.
///
/// A hash whose fields are the declared view names and whose values are each view's newest
/// event timestamp in microseconds, encoded the way [`crate::online::encode_freshness`]
/// encodes a freshness field. One key serves three purposes: the declared view set garbage
/// collection diffs against, a per-view freshness source for metrics, and a signal that a
/// refresh happened.
///
/// The project name is not escaped, because a key is not a pattern. A caller that turns it
/// into one has to escape it itself; [`crate::online::ProjectScan`] is where that happens.
///
/// Args:
///     project: The project name.
///
/// Returns:
///     The registry key.
pub fn views_registry_key(project: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(project.len() + b":views".len());
    out.extend_from_slice(project.as_bytes());
    out.extend_from_slice(b":views");
    out
}

/// The field name holding a view's freshness timestamp.
pub fn freshness_field(view: &str) -> String {
    format!("f:{view}")
}

/// The field name holding a view's encoded feature vector.
///
/// One field per view, not one per feature. The value codec writes a whole
/// vector in a single fixed-stride blob, so a per-feature field name would either
/// duplicate the schema tag per feature or force the codec down to one column at
/// a time, losing the stride. The cost is that reading two features from a
/// 50-feature view decodes 50 columns; the benefit is one field per view in
/// storage and roughly half the bytes.
pub fn value_field(view: &str) -> String {
    format!("v:{view}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_simple_components() {
        let encoded = encode_entity_key(&[b"123", b"abc"]).unwrap();
        assert_eq!(encoded, b"3:123|3:abc");
        assert_eq!(
            decode_entity_key(&encoded).unwrap(),
            vec![b"123".to_vec(), b"abc".to_vec()]
        );
    }

    #[test]
    fn separator_inside_a_value_is_not_ambiguous() {
        // The whole reason for length prefixes.
        let encoded = encode_entity_key(&[b"a|b", b"c"]).unwrap();
        assert_eq!(encoded, b"3:a|b|1:c");
        assert_eq!(
            decode_entity_key(&encoded).unwrap(),
            vec![b"a|b".to_vec(), b"c".to_vec()]
        );
    }

    #[test]
    fn empty_component_is_allowed() {
        let encoded = encode_entity_key(&[b""]).unwrap();
        assert_eq!(encoded, b"0:");
        assert_eq!(decode_entity_key(&encoded).unwrap(), vec![Vec::<u8>::new()]);
    }

    #[test]
    fn arbitrary_bytes_round_trip() {
        let raw = [0u8, 255, b':', b'|', 10];
        let encoded = encode_entity_key(&[&raw]).unwrap();
        assert_eq!(decode_entity_key(&encoded).unwrap(), vec![raw.to_vec()]);
    }

    #[test]
    fn empty_tuple_is_rejected() {
        assert!(matches!(
            encode_entity_key(&[]),
            Err(Error::EmptyEntityKey { .. })
        ));
    }

    #[test]
    fn oversized_component_is_rejected() {
        let big = vec![b'x'; MAX_COMPONENT_LEN + 1];
        assert!(matches!(
            encode_entity_key(&[&big]),
            Err(Error::EntityKeyComponentTooLong { .. })
        ));
    }

    #[test]
    fn at_the_cap_is_accepted() {
        let exact = vec![b'x'; MAX_COMPONENT_LEN];
        assert!(encode_entity_key(&[&exact]).is_ok());
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        for bad in [
            &b""[..],
            b"abc",
            b"3",
            b"3:",
            b"3:ab",
            b"3:abcX",
            b"9999999:x",
            b"2:ab|",
            b"|2:ab",
        ] {
            assert!(
                decode_entity_key(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn hash_key_round_trips() {
        let encoded = encode_entity_key(&[b"u1"]).unwrap();
        let key = entity_hash_key("ads", "user", &encoded);
        assert_eq!(key, b"ads:user:2:u1");
        let (project, entity, rest) = parse_entity_hash_key(&key).unwrap();
        assert_eq!(project, "ads");
        assert_eq!(entity, "user");
        assert_eq!(decode_entity_key(rest).unwrap(), vec![b"u1".to_vec()]);
    }

    #[test]
    fn field_names_are_namespaced() {
        assert_eq!(value_field("clicks"), "v:clicks");
        assert_eq!(freshness_field("clicks"), "f:clicks");
    }

    #[test]
    fn a_registry_key_is_not_an_entity_hash() {
        assert_eq!(views_registry_key("ads"), b"ads:views");
        assert!(!is_entity_hash_key(&views_registry_key("ads"), "ads"));
    }

    #[test]
    fn an_entity_hash_is_recognised_for_its_own_project_only() {
        let encoded = encode_entity_key(&[b"u1"]).unwrap();
        let key = entity_hash_key("ads", "user", &encoded);
        assert!(is_entity_hash_key(&key, "ads"));
        assert!(!is_entity_hash_key(&key, "other"));
    }

    #[test]
    fn a_key_that_only_looks_like_one_is_left_alone() {
        for bad in [
            &b"ads:views:x:y"[..],
            b"ads",
            b"ads:user",
            b"ads:user:2:u1|",
        ] {
            assert!(
                !is_entity_hash_key(bad, "ads"),
                "expected {bad:?} to be refused"
            );
        }
    }

    #[test]
    fn an_integer_key_component_is_its_decimal_form() {
        use arrow::array::{Int32Array, Int64Array};
        let wide = Int64Array::from(vec![Some(123), None]);
        assert_eq!(entity_key_component(&wide, 0).unwrap(), b"123");
        assert!(matches!(
            entity_key_component(&wide, 1),
            Err(Error::NullEntityKey)
        ));

        let narrow = Int32Array::from(vec![7]);
        assert_eq!(entity_key_component(&narrow, 0).unwrap(), b"7");
    }

    #[test]
    fn a_string_key_component_is_its_own_bytes() {
        use arrow::array::StringArray;
        let text = StringArray::from(vec!["a|b", ""]);
        assert_eq!(entity_key_component(&text, 0).unwrap(), b"a|b");
        assert_eq!(entity_key_component(&text, 1).unwrap(), b"");
    }

    #[test]
    fn a_key_type_the_encoder_cannot_carry_is_refused_by_name() {
        use arrow::array::Float64Array;
        let floats = Float64Array::from(vec![1.5]);
        match entity_key_component(&floats, 0) {
            Err(Error::UnsupportedKeyType { dtype }) => assert_eq!(dtype, "Float64"),
            other => panic!("expected an unsupported key type, got {other:?}"),
        }
    }
}
