//! Random phrases for a typing lock: type this back, in full, to unlock.
//!
//! Shared by every typing lock in the app — a block list's, and a settings
//! lock's — so the alphabet, bounds, and generation all agree in one place.

use rand::Rng;

/// Floor on phrase length. Below this a "lock" is trivially typed by accident.
pub const MIN_PHRASE_LENGTH: u32 = 10;
/// Ceiling on phrase length. Above this a typo in the field asks someone to
/// type tens of thousands of characters.
pub const MAX_PHRASE_LENGTH: u32 = 5000;

/// Characters a phrase is drawn from: upper- and lower-case letters and
/// digits. Wide enough that a phrase never looks accidentally patterned,
/// plain enough to type without hunting for punctuation.
const PHRASE_ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Whether `length` falls within the supported range.
pub fn is_valid_length(length: u32) -> bool {
    (MIN_PHRASE_LENGTH..=MAX_PHRASE_LENGTH).contains(&length)
}

/// Generate a fresh random phrase of exactly `length` characters.
pub fn generate(length: u32) -> String {
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| PHRASE_ALPHABET[rng.gen_range(0..PHRASE_ALPHABET.len())] as char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_exactly_the_requested_length() {
        assert_eq!(generate(1).chars().count(), 1);
        assert_eq!(generate(200).chars().count(), 200);
    }

    #[test]
    fn only_draws_from_the_documented_alphabet() {
        let phrase = generate(500);
        assert!(phrase.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn length_bounds_are_inclusive() {
        assert!(is_valid_length(MIN_PHRASE_LENGTH));
        assert!(is_valid_length(MAX_PHRASE_LENGTH));
        assert!(!is_valid_length(MIN_PHRASE_LENGTH - 1));
        assert!(!is_valid_length(MAX_PHRASE_LENGTH + 1));
    }
}
