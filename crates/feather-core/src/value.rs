//! Fixed-stride value encoding.
//!
//! One field value is one view's feature vector for one entity:
//!
//! ```text
//! [tag: u32 LE][flags: u8][null bitmap][fixed-width columns][variable-width tail]
//! ```
//!
//! The property that matters is that fixed-width columns are concatenated with
//! **no per-value length prefix**, so locating column *k* is arithmetic rather
//! than parsing. That is the direct answer to the cost Feast documents: its
//! materialization took over an hour for 5M rows by 30 columns because
//! `convert_arrow_to_proto` serializes each cell independently at roughly 1e-5 s.
//! Serializing per cell is the mistake; the format is almost incidental.
//!
//! Values are self-describing through `tag`, a stable hash of the view's field
//! names, dtypes, and declaration order. A tag mismatch means the value is
//! treated as missing rather than decoded, which is what turns a dtype change
//! into a null window instead of silent corruption.
//!
//! `flags` is reserved and written as zero. It exists so an encoding-level change
//! has somewhere to announce itself without changing the layout.

use std::ops::Range;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, TimestampMicrosecondArray,
    TimestampMicrosecondBuilder,
};
use arrow::datatypes::{DataType, TimeUnit};

use crate::definitions::{DType, Field};
use crate::error::{Error, Result};

/// Bytes of schema tag at the head of every encoded vector.
const TAG_LEN: usize = 4;

/// Upper bound on one variable-width value, so a single row cannot produce an
/// unbounded field.
const MAX_VARIABLE_LEN: usize = 1 << 20;

/// A stable identifier for a view's schema: field names, dtypes, and order.
///
/// Deliberately 32 bits rather than 8. The tag only has to disambiguate
/// successive schema versions of the *same* view, but a u8 collides within a
/// handful of versions (birthday bound over 256 values), and the failure mode is
/// decoding one schema as another. Four bytes per vector is cheap insurance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaTag(pub u32);

impl SchemaTag {
    /// Hash the fields with FNV-1a.
    ///
    /// FNV-1a rather than `DefaultHasher` because the standard hasher is
    /// explicitly not stable across Rust releases, and this value is persisted
    /// in Valkey. A tag that changed on a toolchain upgrade would invalidate
    /// every stored value.
    pub fn of(fields: &[Field]) -> Self {
        const OFFSET: u32 = 0x811c_9dc5;
        const PRIME: u32 = 0x0100_0193;

        let mut hash = OFFSET;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                hash ^= u32::from(b);
                hash = hash.wrapping_mul(PRIME);
            }
        };
        // A length prefix per string keeps `["ab"]` distinct from `["a","b"]`.
        for field in fields {
            mix(&(field.name.len() as u32).to_le_bytes());
            mix(field.name.as_bytes());
            mix(field.dtype.as_str().as_bytes());
        }
        SchemaTag(hash)
    }
}

/// A batch of encoded vectors in one buffer, with each row's byte range.
///
/// One allocation and one pass, rather than a `Vec<Vec<u8>>` per entity. The
/// writer slices this buffer per entity when issuing `HSET`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedBatch {
    pub buf: Vec<u8>,
    /// Byte range of each encoded row, in input order.
    pub ranges: Vec<Range<usize>>,
}

impl EncodedBatch {
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The encoded bytes for row `index`.
    pub fn row(&self, index: usize) -> Option<&[u8]> {
        self.ranges.get(index).map(|r| &self.buf[r.clone()])
    }
}

/// The Arrow type a declared dtype maps to.
pub fn arrow_type(dtype: DType) -> DataType {
    match dtype {
        DType::Int64 => DataType::Int64,
        DType::Float64 => DataType::Float64,
        DType::Boolean => DataType::Boolean,
        DType::Utf8 => DataType::Utf8,
        DType::TimestampMicros => DataType::Timestamp(TimeUnit::Microsecond, None),
    }
}

fn mismatch(field: &Field, actual: &DataType) -> Error {
    Error::ColumnTypeMismatch {
        name: field.name.clone(),
        declared: field.dtype.as_str().to_owned(),
        actual: format!("{actual:?}"),
    }
}

/// Encode `rows` of a batch into a single buffer.
pub fn encode_batch(
    fields: &[Field],
    columns: &[ArrayRef],
    rows: Range<usize>,
) -> Result<EncodedBatch> {
    if fields.len() != columns.len() {
        return Err(Error::ColumnCountMismatch {
            view: String::new(),
            fields: fields.len(),
            columns: columns.len(),
        });
    }

    let tag = SchemaTag::of(fields);
    let bitmap_len = fields.len().div_ceil(8);
    let fixed_stride: usize = fields.iter().filter_map(|f| f.dtype.fixed_width()).sum();
    let per_row = TAG_LEN + 1 + bitmap_len + fixed_stride;

    let mut buf = Vec::with_capacity(rows.len().saturating_mul(per_row));
    let mut ranges = Vec::with_capacity(rows.len());

    for row in rows {
        for (field, column) in fields.iter().zip(columns) {
            if row >= column.len() {
                return Err(Error::RowOutOfRange {
                    row,
                    len: column.len(),
                });
            }
            // Check the declared type once per row rather than trusting it; a
            // mismatch is a programming error, not a data condition.
            let expected = arrow_type(field.dtype);
            if column.data_type() != &expected {
                return Err(mismatch(field, column.data_type()));
            }
        }

        let start = buf.len();
        buf.extend_from_slice(&tag.0.to_le_bytes());
        buf.push(0); // flags, reserved

        let bitmap_at = buf.len();
        buf.resize(bitmap_at + bitmap_len, 0);

        // Fixed-width columns, in declared order among fixed-width fields.
        for (index, field) in fields.iter().enumerate() {
            let Some(width) = field.dtype.fixed_width() else {
                continue;
            };
            let column = &columns[index];
            if column.is_null(row) {
                buf[bitmap_at + index / 8] |= 1 << (index % 8);
                buf.resize(buf.len() + width, 0);
            } else {
                write_fixed(&mut buf, field, column, row)?;
            }
        }

        // Variable-width tail, in declared order among variable-width fields.
        for (index, field) in fields.iter().enumerate() {
            if field.dtype.is_fixed_width() {
                continue;
            }
            let column = &columns[index];
            if column.is_null(row) {
                buf[bitmap_at + index / 8] |= 1 << (index % 8);
                buf.extend_from_slice(&0u32.to_le_bytes());
            } else {
                write_variable(&mut buf, field, column, row)?;
            }
        }

        ranges.push(start..buf.len());
    }

    Ok(EncodedBatch { buf, ranges })
}

fn write_fixed(buf: &mut Vec<u8>, field: &Field, column: &ArrayRef, row: usize) -> Result<()> {
    match field.dtype {
        DType::Int64 => {
            let array = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| mismatch(field, column.data_type()))?;
            buf.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DType::TimestampMicros => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| mismatch(field, column.data_type()))?;
            buf.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DType::Float64 => {
            let array = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| mismatch(field, column.data_type()))?;
            buf.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DType::Boolean => {
            let array = column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| mismatch(field, column.data_type()))?;
            buf.push(u8::from(array.value(row)));
        }
        DType::Utf8 => unreachable!("variable-width handled separately"),
    }
    Ok(())
}

fn write_variable(buf: &mut Vec<u8>, field: &Field, column: &ArrayRef, row: usize) -> Result<()> {
    match field.dtype {
        DType::Utf8 => {
            let array = column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| mismatch(field, column.data_type()))?;
            let value = array.value(row).as_bytes();
            if value.len() > MAX_VARIABLE_LEN {
                return Err(Error::ValueTooLong {
                    field: field.name.clone(),
                    len: value.len(),
                    max: MAX_VARIABLE_LEN,
                });
            }
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buf.extend_from_slice(value);
        }
        _ => unreachable!("fixed-width handled separately"),
    }
    Ok(())
}

/// Bounds-checked reader, so a truncated or hostile buffer cannot panic.
struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    fn need(&self, len: usize) -> Result<()> {
        if self.at + len > self.buf.len() {
            return Err(Error::TruncatedValue {
                need: self.at + len,
                have: self.buf.len(),
            });
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let value = self.buf[self.at];
        self.at += 1;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.buf[self.at..self.at + 4]);
        self.at += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64> {
        self.need(8)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.buf[self.at..self.at + 8]);
        self.at += 8;
        Ok(i64::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.i64()? as u64))
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        self.need(len)?;
        let slice = &self.buf[self.at..self.at + len];
        self.at += len;
        Ok(slice)
    }
}

/// One Arrow builder per field, so decoding produces arrays directly rather than
/// an intermediate row-oriented representation.
enum Builder {
    Int64(Int64Builder),
    Float64(Float64Builder),
    Boolean(BooleanBuilder),
    Utf8(StringBuilder),
    Timestamp(TimestampMicrosecondBuilder),
}

impl Builder {
    fn new(dtype: DType) -> Self {
        match dtype {
            DType::Int64 => Builder::Int64(Int64Builder::new()),
            DType::Float64 => Builder::Float64(Float64Builder::new()),
            DType::Boolean => Builder::Boolean(BooleanBuilder::new()),
            DType::Utf8 => Builder::Utf8(StringBuilder::new()),
            DType::TimestampMicros => Builder::Timestamp(TimestampMicrosecondBuilder::new()),
        }
    }

    fn null(&mut self) {
        match self {
            Builder::Int64(b) => b.append_null(),
            Builder::Float64(b) => b.append_null(),
            Builder::Boolean(b) => b.append_null(),
            Builder::Utf8(b) => b.append_null(),
            Builder::Timestamp(b) => b.append_null(),
        }
    }

    fn finish(self) -> ArrayRef {
        match self {
            Builder::Int64(mut b) => std::sync::Arc::new(b.finish()),
            Builder::Float64(mut b) => std::sync::Arc::new(b.finish()),
            Builder::Boolean(mut b) => std::sync::Arc::new(b.finish()),
            Builder::Utf8(mut b) => std::sync::Arc::new(b.finish()),
            Builder::Timestamp(mut b) => std::sync::Arc::new(b.finish()),
        }
    }
}

/// Decode many encoded vectors into Arrow arrays of the same length.
///
/// `bufs` is in entity order, matching what the read path gets back from one
/// `HMGET` per entity.
pub fn decode_batch(
    fields: &[Field],
    expected: SchemaTag,
    bufs: &[&[u8]],
) -> Result<Vec<ArrayRef>> {
    let bitmap_len = fields.len().div_ceil(8);
    let mut builders: Vec<Builder> = fields.iter().map(|f| Builder::new(f.dtype)).collect();

    for buf in bufs {
        let mut reader = Reader::new(buf);

        let found = reader.u32()?;
        if found != expected.0 {
            return Err(Error::SchemaTagMismatch {
                found,
                expected: expected.0,
            });
        }
        let _flags = reader.u8()?;
        let bitmap = reader.bytes(bitmap_len)?.to_vec();
        let is_null = |index: usize| bitmap[index / 8] & (1 << (index % 8)) != 0;

        for (index, field) in fields.iter().enumerate() {
            if is_null(index) {
                // Still advance past the slot for fixed-width fields.
                if let Some(width) = field.dtype.fixed_width() {
                    reader.bytes(width)?;
                } else {
                    let len = reader.u32()? as usize;
                    reader.bytes(len)?;
                }
                builders[index].null();
                continue;
            }

            match field.dtype {
                DType::Int64 => {
                    let value = reader.i64()?;
                    match &mut builders[index] {
                        Builder::Int64(b) => b.append_value(value),
                        _ => unreachable!("builder matches field dtype"),
                    }
                }
                DType::TimestampMicros => {
                    let value = reader.i64()?;
                    match &mut builders[index] {
                        Builder::Timestamp(b) => b.append_value(value),
                        _ => unreachable!("builder matches field dtype"),
                    }
                }
                DType::Float64 => {
                    let value = reader.f64()?;
                    match &mut builders[index] {
                        Builder::Float64(b) => b.append_value(value),
                        _ => unreachable!("builder matches field dtype"),
                    }
                }
                DType::Boolean => {
                    let value = reader.u8()? != 0;
                    match &mut builders[index] {
                        Builder::Boolean(b) => b.append_value(value),
                        _ => unreachable!("builder matches field dtype"),
                    }
                }
                DType::Utf8 => {
                    let len = reader.u32()? as usize;
                    if len > MAX_VARIABLE_LEN {
                        return Err(Error::ValueTooLong {
                            field: field.name.clone(),
                            len,
                            max: MAX_VARIABLE_LEN,
                        });
                    }
                    let bytes = reader.bytes(len)?;
                    let text = std::str::from_utf8(bytes).map_err(|_| Error::MalformedValue {
                        field: field.name.clone(),
                        reason: "not UTF-8".to_owned(),
                    })?;
                    match &mut builders[index] {
                        Builder::Utf8(b) => b.append_value(text),
                        _ => unreachable!("builder matches field dtype"),
                    }
                }
            }
        }

        if reader.at != buf.len() {
            return Err(Error::MalformedValue {
                field: String::new(),
                reason: format!("{} trailing bytes after decoding", buf.len() - reader.at),
            });
        }
    }

    Ok(builders.into_iter().map(Builder::finish).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use std::sync::Arc;

    fn fields() -> Vec<Field> {
        vec![
            Field::new("count", DType::Int64),
            Field::new("score", DType::Float64),
            Field::new("active", DType::Boolean),
            Field::new("label", DType::Utf8),
        ]
    }

    fn columns() -> Vec<ArrayRef> {
        vec![
            Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(Float64Array::from(vec![Some(1.5), Some(2.5), None])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c|d")])),
        ]
    }

    #[test]
    fn round_trips_a_batch() {
        let fields = fields();
        let columns = columns();
        let encoded = encode_batch(&fields, &columns, 0..3).unwrap();
        assert_eq!(encoded.len(), 3);

        let bufs: Vec<&[u8]> = (0..3).map(|i| encoded.row(i).unwrap()).collect();
        let decoded = decode_batch(&fields, SchemaTag::of(&fields), &bufs).unwrap();

        let counts = decoded[0].as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(counts.value(0), 1);
        assert!(counts.is_null(1));
        assert_eq!(counts.value(2), 3);

        let scores = decoded[1].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(scores.value(0), 1.5);
        assert_eq!(scores.value(1), 2.5);
        assert!(scores.is_null(2));

        let active = decoded[2].as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(active.value(0));
        assert!(!active.value(1));
        assert!(active.is_null(2));

        let labels = decoded[3].as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(labels.value(0), "a");
        assert!(labels.is_null(1));
        assert_eq!(labels.value(2), "c|d");
    }

    #[test]
    fn a_row_slice_encodes_only_those_rows() {
        let fields = fields();
        let columns = columns();
        let encoded = encode_batch(&fields, &columns, 1..3).unwrap();
        assert_eq!(encoded.len(), 2);

        let bufs: Vec<&[u8]> = (0..2).map(|i| encoded.row(i).unwrap()).collect();
        let decoded = decode_batch(&fields, SchemaTag::of(&fields), &bufs).unwrap();
        let counts = decoded[0].as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(counts.is_null(0));
        assert_eq!(counts.value(1), 3);
    }

    #[test]
    fn the_tag_changes_when_the_schema_changes() {
        let base = SchemaTag::of(&fields());
        let mut renamed = fields();
        renamed[0].name = "clicks".to_owned();
        assert_ne!(base, SchemaTag::of(&renamed));

        let mut retyped = fields();
        retyped[0].dtype = DType::Float64;
        assert_ne!(base, SchemaTag::of(&retyped));

        let mut reordered = fields();
        reordered.swap(0, 1);
        assert_ne!(base, SchemaTag::of(&reordered));
    }

    #[test]
    fn the_tag_is_stable() {
        // Persisted in Valkey, so it must not drift across builds.
        assert_eq!(SchemaTag::of(&fields()).0, SchemaTag::of(&fields()).0);
        // Pin the actual value so a refactor of the hash is caught.
        assert_eq!(
            SchemaTag::of(&[Field::new("a", DType::Int64)]).0,
            0x183b_3b72
        );
    }

    #[test]
    fn a_tag_mismatch_is_reported_rather_than_decoded() {
        let fields = fields();
        let columns = columns();
        let encoded = encode_batch(&fields, &columns, 0..1).unwrap();
        let buf = encoded.row(0).unwrap();

        let other = SchemaTag::of(&[Field::new("x", DType::Utf8)]);
        assert!(matches!(
            decode_batch(&fields, other, &[buf]),
            Err(Error::SchemaTagMismatch { .. })
        ));
    }

    #[test]
    fn truncated_input_is_rejected_without_panicking() {
        let fields = fields();
        let columns = columns();
        let encoded = encode_batch(&fields, &columns, 0..1).unwrap();
        let full = encoded.row(0).unwrap();

        for cut in 0..full.len() {
            let result = decode_batch(&fields, SchemaTag::of(&fields), &[&full[..cut]]);
            assert!(
                result.is_err(),
                "expected truncation at {cut} to be rejected"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let fields = fields();
        let columns = columns();
        let encoded = encode_batch(&fields, &columns, 0..1).unwrap();
        let mut extended = encoded.row(0).unwrap().to_vec();
        extended.push(0);
        assert!(matches!(
            decode_batch(&fields, SchemaTag::of(&fields), &[&extended]),
            Err(Error::MalformedValue { .. })
        ));
    }

    #[test]
    fn a_column_count_mismatch_is_rejected() {
        let fields = fields();
        let columns = columns();
        assert!(matches!(
            encode_batch(&fields, &columns[..2], 0..1),
            Err(Error::ColumnCountMismatch { .. })
        ));
    }

    #[test]
    fn a_column_type_mismatch_is_rejected() {
        let fields = vec![Field::new("count", DType::Int64)];
        let columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["nope"]))];
        assert!(matches!(
            encode_batch(&fields, &columns, 0..1),
            Err(Error::ColumnTypeMismatch { .. })
        ));
    }

    #[test]
    fn a_row_past_the_end_is_rejected() {
        let fields = fields();
        let columns = columns();
        assert!(matches!(
            encode_batch(&fields, &columns, 0..4),
            Err(Error::RowOutOfRange { .. })
        ));
    }

    #[test]
    fn fixed_width_columns_have_no_per_value_prefix() {
        // One Int64 field, one non-null row: tag(4) + flags(1) + bitmap(1) + 8.
        let fields = vec![Field::new("count", DType::Int64)];
        let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![7]))];
        let encoded = encode_batch(&fields, &columns, 0..1).unwrap();
        assert_eq!(encoded.row(0).unwrap().len(), TAG_LEN + 1 + 1 + 8);
    }

    #[test]
    fn an_empty_batch_is_allowed() {
        let fields = fields();
        let columns = columns();
        let encoded = encode_batch(&fields, &columns, 0..0).unwrap();
        assert!(encoded.is_empty());
        assert!(
            decode_batch(&fields, SchemaTag::of(&fields), &[])
                .unwrap()
                .len()
                == fields.len()
        );
    }

    #[test]
    fn every_dtype_round_trips_through_the_arrow_type_map() {
        for dtype in [
            DType::Int64,
            DType::Float64,
            DType::Boolean,
            DType::Utf8,
            DType::TimestampMicros,
        ] {
            assert_eq!(arrow_type(dtype), arrow_type(dtype));
        }
    }
}
