use std::hash::Hash;

use rustc_hash::FxHashMap as HashMap;

pub struct Counter<T: Hash>(HashMap<T, usize>);
impl<T: Hash> Default for Counter<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<T: Hash + std::cmp::Eq> Counter<T> {
    pub fn keys(&self) -> impl Iterator<Item = &T> {
        self.0.keys()
    }
    pub fn insert(&mut self, x: T) {
        self.increment(x, 1);
    }
    pub fn increment(&mut self, x: T, inc: usize) {
        *self.0.entry(x).or_default() += inc;
    }
    pub fn most_common(&self) -> Option<&T> {
        self.0.iter().max_by_key(|(_, freq)| **freq).map(|(k, _)| k)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&T, &usize)> {
        self.0.iter()
    }
}
