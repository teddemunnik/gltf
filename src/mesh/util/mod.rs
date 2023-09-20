/// Casting iterator adapters for colors.
pub mod colors;

/// Casting iterator adapters for vertex indices.
pub mod indices;

/// Casting iterator adapters for joint indices.
pub mod joints;

/// Casting iterator adapters for texture co-ordinates.
pub mod tex_coords;

/// Casting iterator adapters for node weights.
pub mod weights;

use crate::mesh;

use crate::accessor::Iter;
use crate::Buffer;

/// XYZ vertex positions of type `[f32; 3]`.
pub type ReadPositions<'a> = Iter<'a, [f32; 3]>;

/// Directly reads positions which may have possibly been stored in a quantized format, without performing internal conversions to floating point.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ReadPositionsDirect<'a>{
    /// XYX vertex position stored as floating point 
    Float(Iter<'a, [f32; 3]>),

    /// XYZ vertex possitions stored as signed bytes. Should possibly be treated as normalized [-1..1] values.
    I8(Iter<'a, [i8; 3]>),

    /// XYZ vertex possitions stored as signed bytes. Should possibly be treated as normalized [-1..1] values.
    NormalizedI8(Iter<'a, [i8; 3]>),

    /// TODO
    U8(Iter<'a, [u8; 3]>),

    /// TODO
    NormalizedU8(Iter<'a, [u8; 3]>),

    /// XYZ vertex possitions stored as signed shorts. Should possibly be treated as normalized [-1..1] values.
    I16(Iter<'a, [i16; 3]>),

    /// XYZ vertex possitions stored as signed shorts. Should possibly be treated as normalized [0..1] values.
    NormalizedI16(Iter<'a, [i16; 3]>),

    /// TODO
    U16(Iter<'a, [u16; 3]>),

    /// TODO
    NormalizedU16(Iter<'a, [u16; 3]>)
}

/// XYZ vertex normals of type `[f32; 3]`.
pub type ReadNormals<'a> = Iter<'a, [f32; 3]>;

/// Directly reads normals which may have possibly been stored in a quantized format, without performing internal conversions to floating point.
#[non_exhaustive]
pub enum ReadNormalsDirect<'a>{

    /// XYZ vertex normal stored as bytes. Should be treated as normalized [-1..1] values.
    NormalizedI8(Iter<'a, [i8; 3]>),

    /// XYZ vertex normal stored as shorts. Should be treated as normalized [-1..1] values.
    NormalizedI16(Iter<'a, [i16; 3]>),

    /// XYZ vertex normal stored as normalized floating point
    F32(Iter<'a, [f32; 3]>),
}

/// TODO
pub struct NormalsAsF32<'a>{
    iter: ReadNormalsDirect<'a>
}

fn normalized_i16_to_float(x: [i16; 3]) -> [f32; 3]{
    [
        ((x[0] as f32) / 32767.0).max(-1.0),
        ((x[1] as f32) / 32767.0).max(-1.0),
        ((x[2] as f32) / 32767.0).max(-1.0)
    ]
}

fn normalized_i8_to_float(x: [i8; 3]) -> [f32; 3]{
    [
        ((x[0] as f32) / 127.0).max(-1.0),
        ((x[1] as f32) / 127.0).max(-1.0),
        ((x[2] as f32) / 127.0).max(-1.0)
    ]
}

fn normalized_u16_to_float(x: [u16; 3]) -> [f32; 3]{
    [
        x[0] as f32 / 65535.0,
        x[1] as f32 / 65535.0,
        x[2] as f32 / 65535.0,
    ]
}

fn normalized_u8_to_float(x: [u8; 3]) -> [f32; 3]{
    [
        x[0] as f32 / 255.0,
        x[1] as f32 / 255.0,
        x[2] as f32 / 255.0
    ]
}

impl<'a> Iterator for NormalsAsF32<'a>{
    type Item = [f32; 3];

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter{
            ReadNormalsDirect::F32(ref mut iter) => iter.next(),
            ReadNormalsDirect::NormalizedI16(ref mut iter) => iter.next().map(normalized_i16_to_float),
            ReadNormalsDirect::NormalizedI8(ref mut iter) => iter.next().map(normalized_i8_to_float),
        }
    }
}


/// TODO
pub struct PositionsAsF32<'a>{
    iter: ReadPositionsDirect<'a>
}

impl<'a> Iterator for PositionsAsF32<'a>{
    type Item = [f32; 3];

    fn next(&mut self) -> Option<Self::Item>{
        match self.iter{
            ReadPositionsDirect::Float(ref mut iter) => iter.next(),
            ReadPositionsDirect::I8(ref mut iter) => iter.next().map(|x| [x[0] as f32, x[1] as f32, x[2] as f32]),
            ReadPositionsDirect::NormalizedI8(ref mut iter) => iter.next().map(normalized_i8_to_float),
            ReadPositionsDirect::U8(ref mut iter) => iter.next().map(|x| [x[0] as f32, x[1] as f32, x[2] as f32]),
            ReadPositionsDirect::NormalizedU8(ref mut iter) => iter.next().map(normalized_u8_to_float),
            ReadPositionsDirect::I16(ref mut iter) => iter.next().map(|x| [x[0] as f32, x[1] as f32, x[2] as f32]),
            ReadPositionsDirect::NormalizedI16(ref mut iter) => iter.next().map(normalized_i16_to_float),
            ReadPositionsDirect::U16(ref mut iter) => iter.next().map(|x| [x[0] as f32, x[1] as f32, x[2] as f32]),
            ReadPositionsDirect::NormalizedU16(ref mut iter) => iter.next().map(normalized_u16_to_float),
        }
    }
}

impl<'a> ReadNormalsDirect<'a>{
    /// TODO
    pub fn as_floats(self) -> NormalsAsF32<'a>{
        NormalsAsF32 { iter: self }
    }
}

impl<'a> ReadPositionsDirect<'a>{
    /// TODO
    pub fn as_floats(self) -> PositionsAsF32<'a>{
        PositionsAsF32{ iter: self }
    }
}

/// XYZW vertex tangents of type `[f32; 4]` where the `w` component is a
/// sign value (-1 or +1) indicating the handedness of the tangent basis.
pub type ReadTangents<'a> = Iter<'a, [f32; 4]>;

/// Directly reads normals which may have possibly been stored in a quantized format, without performing internal conversions to floating point.
pub enum ReadTangentsDirect<'a>{
    /// XYZ vertex tangent stored as normalized floating point
    Float(Iter<'a, [f32; 3]>),

    /// XYZ vertex tangent stored as bytes. Should be treated as normalized [-1..1] values.
    I8(Iter<'a, [i8; 3]>),

    /// XYZ vertex tangent stored as shorts. Should be treated as normalized [-1..1] values.
    I16(Iter<'a, [i16; 3]>),
}

/// XYZ vertex position displacements of type `[f32; 3]`.
pub type ReadPositionDisplacements<'a> = Iter<'a, [f32; 3]>;

/// XYZ vertex normal displacements of type `[f32; 3]`.
pub type ReadNormalDisplacements<'a> = Iter<'a, [f32; 3]>;

/// XYZ vertex tangent displacements.
pub type ReadTangentDisplacements<'a> = Iter<'a, [f32; 3]>;

/// Vertex colors.
#[derive(Clone, Debug)]
pub enum ReadColors<'a> {
    /// RGB vertex color of type `[u8; 3]>`.
    RgbU8(Iter<'a, [u8; 3]>),
    /// RGB vertex color of type `[u16; 3]>`.
    RgbU16(Iter<'a, [u16; 3]>),
    /// RGB vertex color of type `[f32; 3]`.
    RgbF32(Iter<'a, [f32; 3]>),
    /// RGBA vertex color of type `[u8; 4]>`.
    RgbaU8(Iter<'a, [u8; 4]>),
    /// RGBA vertex color of type `[u16; 4]>`.
    RgbaU16(Iter<'a, [u16; 4]>),
    /// RGBA vertex color of type `[f32; 4]`.
    RgbaF32(Iter<'a, [f32; 4]>),
}

/// Index data.
#[derive(Clone, Debug)]
pub enum ReadIndices<'a> {
    /// Index data of type U8
    U8(Iter<'a, u8>),
    /// Index data of type U16
    U16(Iter<'a, u16>),
    /// Index data of type U32
    U32(Iter<'a, u32>),
}

/// Vertex joints.
#[derive(Clone, Debug)]
pub enum ReadJoints<'a> {
    /// Joints of type `[u8; 4]`.
    /// Refer to the documentation on morph targets and skins for more
    /// information.
    U8(Iter<'a, [u8; 4]>),
    /// Joints of type `[u16; 4]`.
    /// Refer to the documentation on morph targets and skins for more
    /// information.
    U16(Iter<'a, [u16; 4]>),
}

/// UV texture co-ordinates.
#[derive(Clone, Debug)]
pub enum ReadTexCoords<'a> {
    /// UV texture co-ordinates of type `[u8; 2]>`.
    U8(Iter<'a, [u8; 2]>),
    /// UV texture co-ordinates of type `[u16; 2]>`.
    U16(Iter<'a, [u16; 2]>),
    /// UV texture co-ordinates of type `[f32; 2]`.
    F32(Iter<'a, [f32; 2]>),
}


/// TODO
#[non_exhaustive]
pub enum ReadTexCoordsExtended<'a>{
    /// UV texture co-ordinates of type `[i8; 2]>`.
    I8(bool, Iter<'a, [i8; 2]>),
    /// UV texture co-ordinates of type `[u8; 2]>`.
    U8(bool, Iter<'a, [u8; 2]>),
    /// UV texture co-ordinates of type `[i16; 2]>`.
    I16(bool, Iter<'a, [i16; 2]>),
    /// UV texture co-ordinates of type `[u16; 2]>`.
    U16(bool, Iter<'a, [u16; 2]>),
    /// UV texture co-ordinates of type `[f32; 2]`.
    F32(Iter<'a, [f32; 2]>),
}

/// Weights.
#[derive(Clone, Debug)]
pub enum ReadWeights<'a> {
    /// Weights of type `[u8; 4]`.
    U8(Iter<'a, [u8; 4]>),
    /// Weights of type `[u16; 4]`.
    U16(Iter<'a, [u16; 4]>),
    /// Weights of type `[f32; 4]`.
    F32(Iter<'a, [f32; 4]>),
}

/// Morph targets.
#[derive(Clone, Debug)]
pub struct ReadMorphTargets<'a, 's, F>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    pub(crate) index: usize,
    pub(crate) reader: mesh::Reader<'a, 's, F>,
}

impl<'a, 's, F> ExactSizeIterator for ReadMorphTargets<'a, 's, F> where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>
{
}

impl<'a, 's, F> Iterator for ReadMorphTargets<'a, 's, F>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    type Item = (
        Option<ReadPositionDisplacements<'s>>,
        Option<ReadNormalDisplacements<'s>>,
        Option<ReadTangentDisplacements<'s>>,
    );
    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        self.reader
            .primitive
            .morph_targets()
            .nth(self.index - 1)
            .map(|morph_target| {
                let positions = morph_target
                    .positions()
                    .and_then(|accessor| Iter::new(accessor, self.reader.get_buffer_data.clone()));
                let normals = morph_target
                    .normals()
                    .and_then(|accessor| Iter::new(accessor, self.reader.get_buffer_data.clone()));
                let tangents = morph_target
                    .tangents()
                    .and_then(|accessor| Iter::new(accessor, self.reader.get_buffer_data.clone()));
                (positions, normals, tangents)
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.reader.primitive.morph_targets().size_hint()
    }
}

impl<'a> ReadColors<'a> {
    /// Reinterpret colors as RGB u8, discarding alpha, if present.  Lossy if
    /// the underlying iterator yields u16, f32 or any RGBA.
    pub fn into_rgb_u8(self) -> self::colors::CastingIter<'a, self::colors::RgbU8> {
        self::colors::CastingIter::new(self)
    }

    /// Reinterpret colors as RGB u16, discarding alpha, if present.  Lossy if
    /// the underlying iterator yields f32 or any RGBA.
    pub fn into_rgb_u16(self) -> self::colors::CastingIter<'a, self::colors::RgbU16> {
        self::colors::CastingIter::new(self)
    }

    /// Reinterpret colors as RGB f32, discarding alpha, if present.  Lossy if
    /// the underlying iterator yields u16 or any RGBA.
    pub fn into_rgb_f32(self) -> self::colors::CastingIter<'a, self::colors::RgbF32> {
        self::colors::CastingIter::new(self)
    }

    /// Reinterpret colors as RGBA u8, with default alpha 255.  Lossy if the
    /// underlying iterator yields u16 or f32.
    pub fn into_rgba_u8(self) -> self::colors::CastingIter<'a, self::colors::RgbaU8> {
        self::colors::CastingIter::new(self)
    }

    /// Reinterpret colors as RGBA u16, with default alpha 65535.  Lossy if the
    /// underlying iterator yields f32.
    pub fn into_rgba_u16(self) -> self::colors::CastingIter<'a, self::colors::RgbaU16> {
        self::colors::CastingIter::new(self)
    }

    /// Reinterpret colors as RGBA f32, with default alpha 1.0.  Lossy if the
    /// underlying iterator yields u16.
    pub fn into_rgba_f32(self) -> self::colors::CastingIter<'a, self::colors::RgbaF32> {
        self::colors::CastingIter::new(self)
    }
}

impl<'a> ReadIndices<'a> {
    /// Reinterpret indices as u32, which can fit any possible index.
    pub fn into_u32(self) -> self::indices::CastingIter<'a, self::indices::U32> {
        self::indices::CastingIter::new(self)
    }
}

impl<'a> ReadJoints<'a> {
    /// Reinterpret joints as u16, which can fit any possible joint.
    pub fn into_u16(self) -> self::joints::CastingIter<'a, self::joints::U16> {
        self::joints::CastingIter::new(self)
    }
}

impl<'a> ReadTexCoords<'a> {
    /// Reinterpret texture coordinates as u8.  Lossy if the underlying iterator
    /// yields u16 or f32.
    pub fn into_u8(self) -> self::tex_coords::CastingIter<'a, self::tex_coords::U8> {
        self::tex_coords::CastingIter::new(self)
    }

    /// Reinterpret texture coordinates as u16.  Lossy if the underlying
    /// iterator yields f32.
    pub fn into_u16(self) -> self::tex_coords::CastingIter<'a, self::tex_coords::U16> {
        self::tex_coords::CastingIter::new(self)
    }

    /// Reinterpret texture coordinates as f32.  Lossy if the underlying
    /// iterator yields u16.
    pub fn into_f32(self) -> self::tex_coords::CastingIter<'a, self::tex_coords::F32> {
        self::tex_coords::CastingIter::new(self)
    }
}

impl<'a> ReadWeights<'a> {
    /// Reinterpret weights as u8.  Lossy if the underlying iterator yields u16
    /// or f32.
    pub fn into_u8(self) -> self::weights::CastingIter<'a, self::weights::U8> {
        self::weights::CastingIter::new(self)
    }

    /// Reinterpret weights as u16.  Lossy if the underlying iterator yields
    /// f32.
    pub fn into_u16(self) -> self::weights::CastingIter<'a, self::weights::U16> {
        self::weights::CastingIter::new(self)
    }

    /// Reinterpret weights as f32.  Lossy if the underlying iterator yields
    /// u16.
    pub fn into_f32(self) -> self::weights::CastingIter<'a, self::weights::F32> {
        self::weights::CastingIter::new(self)
    }
}
