use std::{
    collections::BTreeMap,
    io::{Read, Seek},
};

use byteorder::{LittleEndian, ReadBytesExt};
use num_enum::TryFromPrimitive;

use crate::{
    errors::RlmError,
    tensor::{Tensor, TensorF32, TensorQ4_0, TensorQ8_0},
};

const DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Eq, PartialEq, TryFromPrimitive, Clone)]
#[num_enum(error_type(name = RlmError, constructor = RlmError::InvalidGgmlType))]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    Count = 19,
}

impl GgmlType {
    fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let e_num = reader.read_u32::<LittleEndian>()?;
        let e = Self::try_from(e_num)?;

        Ok(e)
    }
}

#[derive(Debug, Eq, PartialEq, TryFromPrimitive, Clone)]
#[num_enum(error_type(name = RlmError, constructor = RlmError::InvalidGgufMetadataValueType))]
#[repr(u32)]
pub enum GgufMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetadataValueType {
    fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let e_num = reader.read_u32::<LittleEndian>()?;
        let e = Self::try_from(e_num)?;

        Ok(e)
    }
}

#[derive(Debug)]
pub struct GgufString {
    len: u64,
    data: Vec<u8>,
}

impl GgufString {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let len = reader.read_u64::<LittleEndian>()?;
        let mut data = vec![0; len as usize];
        reader.read_exact(&mut data)?;
        Ok(Self { len, data })
    }
}

impl ToString for GgufString {
    fn to_string(&self) -> String {
        String::from_utf8_lossy(&self.data).to_string()
    }
}

#[derive(Debug)]
pub enum GgufArray {
    // typ: GgufMetadataValueType,
    // len: u64,
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Uint16(Vec<u16>),
    Int16(Vec<i16>),
    Uint32(Vec<u32>),
    Int32(Vec<i32>),
    Float(Vec<f32>),
    Uint64(Vec<u64>),
    Int64(Vec<i64>),
    Double(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<GgufString>),
    Array(Vec<GgufArray>),
}

impl GgufArray {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let typ = GgufMetadataValueType::from_reader(&mut reader)?;
        let len = reader.read_u64::<LittleEndian>()?;

        // TODO: refactor with macro
        let data = match typ {
            GgufMetadataValueType::Uint8 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_u8()?);
                }
                Self::Uint8(values)
            }
            GgufMetadataValueType::Int8 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_i8()?);
                }
                Self::Int8(values)
            }
            GgufMetadataValueType::Uint16 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_u16::<LittleEndian>()?);
                }
                Self::Uint16(values)
            }
            GgufMetadataValueType::Int16 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_i16::<LittleEndian>()?);
                }
                Self::Int16(values)
            }
            GgufMetadataValueType::Uint32 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_u32::<LittleEndian>()?);
                }
                Self::Uint32(values)
            }
            GgufMetadataValueType::Int32 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_i32::<LittleEndian>()?);
                }
                Self::Int32(values)
            }
            GgufMetadataValueType::Float32 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_f32::<LittleEndian>()?);
                }
                Self::Float(values)
            }
            GgufMetadataValueType::Bool => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_u8()? != 0);
                }
                Self::Bool(values)
            }
            GgufMetadataValueType::String => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(GgufString::from_reader(&mut reader)?);
                }
                Self::String(values)
            }
            GgufMetadataValueType::Array => {
                return Err(RlmError::Other("not support nest array".to_string()));
                // let mut values = Vec::new();
                // for _ in 0..len {
                //     values.push(GgufArray::from_reader(&mut reader)?);
                // }
                // Self::Array(values)
            }
            GgufMetadataValueType::Uint64 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_u32::<LittleEndian>()?);
                }
                Self::Uint32(values)
            }
            GgufMetadataValueType::Int64 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_i64::<LittleEndian>()?);
                }
                Self::Int64(values)
            }
            GgufMetadataValueType::Float64 => {
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(reader.read_f64::<LittleEndian>()?);
                }
                Self::Double(values)
            }
        };

        Ok(data)
    }
}

#[derive(Debug)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float(f32),
    Uint64(u64),
    Int64(i64),
    Double(f64),
    Bool(bool),
    String(GgufString),
    Array(GgufArray),
}

impl GgufMetadataValue {
    pub fn from_reader(
        mut reader: impl Read,
        typ: GgufMetadataValueType,
    ) -> Result<Self, RlmError> {
        let v = match typ {
            GgufMetadataValueType::Uint8 => Self::Uint8(reader.read_u8()?),
            GgufMetadataValueType::Int8 => Self::Int8(reader.read_i8()?),
            GgufMetadataValueType::Uint16 => Self::Uint16(reader.read_u16::<LittleEndian>()?),
            GgufMetadataValueType::Int16 => Self::Int16(reader.read_i16::<LittleEndian>()?),
            GgufMetadataValueType::Uint32 => Self::Uint32(reader.read_u32::<LittleEndian>()?),
            GgufMetadataValueType::Int32 => Self::Int32(reader.read_i32::<LittleEndian>()?),
            GgufMetadataValueType::Float32 => Self::Float(reader.read_f32::<LittleEndian>()?),
            GgufMetadataValueType::Bool => Self::Bool(reader.read_u8()? != 0),
            GgufMetadataValueType::String => Self::String(GgufString::from_reader(reader)?),
            GgufMetadataValueType::Array => Self::Array(GgufArray::from_reader(reader)?),
            GgufMetadataValueType::Uint64 => Self::Uint64(reader.read_u64::<LittleEndian>()?),
            GgufMetadataValueType::Int64 => Self::Int64(reader.read_i64::<LittleEndian>()?),
            GgufMetadataValueType::Float64 => Self::Double(reader.read_f64::<LittleEndian>()?),
        };

        Ok(v)
    }

    pub fn as_string(&self) -> Option<String> {
        match self {
            GgufMetadataValue::String(v) => Some(v.to_string()),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufMetadataValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufMetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufMetadataValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            GgufMetadataValue::Array(v) => match v {
                GgufArray::String(sa) => {
                    Some(sa.iter().map(|s| s.to_string()).collect::<Vec<String>>())
                }
                _ => None,
            },
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            GgufMetadataValue::Array(v) => match v {
                GgufArray::Float(sa) => Some(sa.clone()),
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct GgufMetadataKv {
    key: GgufString,
    typ: GgufMetadataValueType,
    value: GgufMetadataValue,
}

impl GgufMetadataKv {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let key = GgufString::from_reader(&mut reader)?;
        let typ = GgufMetadataValueType::from_reader(&mut reader)?;
        let value = GgufMetadataValue::from_reader(reader, typ.clone())?;

        Ok(Self { key, typ, value })
    }
}

#[derive(Debug)]
pub struct Metadata(BTreeMap<String, GgufMetadataValue>);

impl Metadata {
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.0.get(key).map(|v| v.as_string()).unwrap_or(None)
    }

    pub fn get_string_result(&self, key: &str) -> Result<String, RlmError> {
        self.get_string(key)
            .ok_or(RlmError::InvalidGgufMetadataKey(key.to_string()))
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.0.get(key).map(|v| v.as_u32()).unwrap_or(None)
    }

    pub fn get_u32_result(&self, key: &str) -> Result<u32, RlmError> {
        self.get_u32(key)
            .ok_or(RlmError::InvalidGgufMetadataKey(key.to_string()))
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.0.get(key).map(|v| v.as_u64()).unwrap_or(None)
    }

    pub fn get_u64_result(&self, key: &str) -> Result<u64, RlmError> {
        self.get_u64(key)
            .ok_or(RlmError::InvalidGgufMetadataKey(key.to_string()))
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.0.get(key).map(|v| v.as_f32()).unwrap_or(None)
    }

    pub fn get_f32_result(&self, key: &str) -> Result<f32, RlmError> {
        self.get_f32(key)
            .ok_or(RlmError::InvalidGgufMetadataKey(key.to_string()))
    }

    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        self.0.get(key).map(|v| v.as_f32_array()).unwrap_or(None)
    }

    pub fn get_f32_array_result(&self, key: &str) -> Result<Vec<f32>, RlmError> {
        self.get_f32_array(key)
            .ok_or(RlmError::InvalidGgufMetadataKey(key.to_string()))
    }

    pub fn get_string_array(&self, key: &str) -> Option<Vec<String>> {
        self.0.get(key).map(|v| v.as_string_array()).unwrap_or(None)
    }

    pub fn get_string_array_result(&self, key: &str) -> Result<Vec<String>, RlmError> {
        self.get_string_array(key)
            .ok_or(RlmError::InvalidGgufMetadataKey(key.to_string()))
    }
}

#[derive(Debug)]
pub struct GgufHeader {
    magic: u32,
    version: u32,
    tensor_count: u64,
    // metadata_kv_count: u64,
    // metadata_kv: Vec<GgufMetadataKv>,
    metadata: Metadata,
}

impl GgufHeader {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let version = reader.read_u32::<LittleEndian>()?;
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

        let mut metadata_map = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let kv = GgufMetadataKv::from_reader(&mut reader)?;
            metadata_map.insert(kv.key.to_string(), kv.value);
        }

        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata: Metadata(metadata_map),
        })
    }
}

#[derive(Debug)]
pub struct GgufTensorInfo {
    name: GgufString,
    n_dimensions: u32,
    dimensions: Vec<u64>,
    typ: GgmlType,
    offset: u64,
}

impl GgufTensorInfo {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let name = GgufString::from_reader(&mut reader)?;
        let n_dimensions = reader.read_u32::<LittleEndian>()?;
        let mut dimensions = Vec::new();
        for _ in 0..n_dimensions {
            dimensions.push(reader.read_u64::<LittleEndian>()?);
        }
        let typ: GgmlType = GgmlType::from_reader(&mut reader)?;
        let offset = reader.read_u64::<LittleEndian>()?;
        // println!("{} {:?} {:?} {}", name.to_string(), typ, dimensions, offset);

        Ok(Self {
            name,
            n_dimensions,
            dimensions,
            typ,
            offset,
        })
    }
}

#[derive(Debug)]
pub struct GgufFile<R> {
    header: GgufHeader,
    tensor_infos: Vec<GgufTensorInfo>,
    _padding: Vec<u8>,
    // tensor_data: Vec<u8>,
    tensor_data_start: u64,
    reader: R,
}

impl<R: Read + Seek> GgufFile<R> {
    pub fn from_reader(mut reader: R) -> Result<Self, RlmError> {
        let header = GgufHeader::from_reader(&mut reader)?;
        let mut tensor_infos = Vec::new();
        for _ in 0..header.tensor_count {
            tensor_infos.push(GgufTensorInfo::from_reader(&mut reader)?);
        }

        let alignment = header
            .metadata
            .get_u32("general.alignment")
            .unwrap_or(DEFAULT_ALIGNMENT) as u64;
        let position = reader.stream_position()?;
        let padding_size = alignment - position % alignment;
        let mut _padding = vec![0; padding_size as usize];
        reader.read_exact(&mut _padding)?;

        let tensor_data_start = reader.stream_position()?;

        Ok(Self {
            header,
            tensor_infos,
            _padding,
            // TODO: read tensor data
            // tensor_data: Vec::new(),
            tensor_data_start,
            reader,
        })
    }

    pub fn metadata(&self) -> &Metadata {
        &self.header.metadata
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<Tensor, RlmError> {
        let i = self
            .tensor_infos
            .iter()
            .find(|i| i.name.to_string() == name.to_string())
            .ok_or(RlmError::TensorNotFound(name.to_string()))?;

        self.reader
            .seek(std::io::SeekFrom::Start(self.tensor_data_start + i.offset))?;

        let n = i.dimensions.iter().product::<u64>() as usize;

        match &i.typ {
            GgmlType::F32 => {
                let mut t = TensorF32::new(n);
                t.from_reader(&mut self.reader)?;
                Ok(Tensor::F32(t))
            }
            GgmlType::Q4_0 => {
                let mut t = TensorQ4_0::new(n);
                t.from_reader(&mut self.reader)?;
                Ok(Tensor::Q4_0(t))
            }
            GgmlType::Q8_0 => {
                let mut t = TensorQ8_0::new(n);
                t.from_reader(&mut self.reader)?;
                Ok(Tensor::Q8_0(t))
            }
            t => Err(RlmError::GgmlTypeNotSupport(t.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use super::GgufFile;

    #[test]
    fn test_gguf_file() {
        let f = File::open("testdata/gemma2b").unwrap();
        let mut reader = BufReader::new(f);
        let gguf_file = GgufFile::from_reader(&mut reader).unwrap();
        println!(
            "{:?} {:?} {:?} {:?}",
            gguf_file.metadata().get_string("general.architecture"),
            gguf_file.metadata().get_string("general.name"),
            gguf_file.metadata().get_u32("general.quantization_version"),
            gguf_file.metadata().0.keys(),
        );
    }
}
