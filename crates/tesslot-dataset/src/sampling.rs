use crate::{Error, Sample};

/// Deterministic bounded-memory shuffle of complete trajectories, not chunks.
/// This is a buffer shuffle, not a globally uniform permutation of a disk corpus.
pub struct Shuffled<I> {
    input: I,
    buffer: Vec<Sample>,
    capacity: usize,
    state: u64,
    input_ended: bool,
    failed: bool,
}

/// Shuffle at most `buffer_samples` complete trajectories in memory at once.
/// Source errors are surfaced and terminate iteration; no invalid sample is used.
pub fn shuffle_samples<I: Iterator<Item = Result<Sample, Error>>>(
    input: I,
    seed: u64,
    buffer_samples: usize,
) -> Result<Shuffled<I>, Error> {
    if buffer_samples == 0 {
        return Err(Error::Contract(
            "shuffle buffer must hold at least one sample".into(),
        ));
    }
    Ok(Shuffled {
        input,
        buffer: Vec::new(),
        capacity: buffer_samples,
        state: seed,
        input_ended: false,
        failed: false,
    })
}

impl<I: Iterator<Item = Result<Sample, Error>>> Iterator for Shuffled<I> {
    type Item = Result<Sample, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        while !self.input_ended && self.buffer.len() < self.capacity {
            match self.input.next() {
                Some(Ok(s)) => self.buffer.push(s),
                Some(Err(e)) => {
                    self.failed = true;
                    self.buffer.clear();
                    return Some(Err(e));
                }
                None => self.input_ended = true,
            }
        }
        if self.buffer.is_empty() {
            return None;
        }
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        let index = (value % self.buffer.len() as u64) as usize;
        Some(Ok(self.buffer.swap_remove(index)))
    }
}
