/// Checks if a byte represents a whitespace character.
///
/// This function is optimized for performance and checks only for the most common
/// whitespace characters: space, tab, newline, and carriage return.
///
/// # Arguments
///
/// * `byte` - The byte to check.
///
/// # Returns
///
/// `true` if the byte represents a whitespace character, `false` otherwise.
///
/// ```

pub fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

pub fn round(x: f32, decimals: u32) -> f32 {
    let y = 10i32.pow(decimals) as f32;
    (x * y).round() / y
}
