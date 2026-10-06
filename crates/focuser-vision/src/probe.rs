//! A filter that learns from the user's own Show/Hide labels.
//!
//! SigLIP describes every image as 768 numbers. A logistic regression on those
//! numbers, trained on the images a user marked, learns where *their* line is
//! far better than thresholds on a handful of prompts can: held out, at
//! Strict it missed about a third as many images that should have been
//! hidden as the prompts did, while wrongly hiding no more.
//!
//! It trains in milliseconds, entirely on the machine.

use serde::{Deserialize, Serialize};

/// Fewest labels of each kind before the probe is trusted over the prompts.
pub const MIN_PER_CLASS: usize = 5;

/// Embeddings are unit vectors with coordinates around ±0.04; scaled up,
/// plain gradient descent settles in a few hundred steps.
const SCALE: f32 = 10.0;
const EPOCHS: usize = 400;
const LEARNING_RATE: f32 = 0.5;
const L2: f32 = 0.01;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    weights: Vec<f32>,
    bias: f32,
    pub hide_labels: usize,
    pub show_labels: usize,
}

impl Probe {
    /// Train on `(embedding, should_hide)` pairs. `None` until there are at
    /// least [`MIN_PER_CLASS`] of each.
    pub fn train(examples: &[(Vec<f32>, bool)]) -> Option<Self> {
        let hide = examples.iter().filter(|(_, h)| *h).count();
        let show = examples.len() - hide;
        if hide < MIN_PER_CLASS || show < MIN_PER_CLASS {
            return None;
        }
        let dim = examples.first()?.0.len();
        if examples.iter().any(|(e, _)| e.len() != dim) {
            return None;
        }
        // People label far more of one kind than the other; weighting each
        // class to the same total keeps the rarer one from being ignored.
        let weight_hide = show as f32 / hide as f32;
        let total = show as f32 + weight_hide * hide as f32;
        let mut w = vec![0f32; dim];
        let mut b = 0f32;
        for _ in 0..EPOCHS {
            let mut grad_w = vec![0f32; dim];
            let mut grad_b = 0f32;
            for (e, hide) in examples {
                let p = sigmoid(b + SCALE * dot(&w, e));
                let c = if *hide { weight_hide * (p - 1.0) } else { p };
                grad_b += c;
                for (g, x) in grad_w.iter_mut().zip(e) {
                    *g += c * SCALE * x;
                }
            }
            b -= LEARNING_RATE * grad_b / total;
            for (wi, g) in w.iter_mut().zip(&grad_w) {
                *wi -= LEARNING_RATE * (g / total + L2 * *wi);
            }
        }
        Some(Self {
            weights: w,
            bias: b,
            hide_labels: hide,
            show_labels: show,
        })
    }

    /// How likely the user would want this image hidden, 0–1.
    pub fn probability(&self, embedding: &[f32]) -> f32 {
        if embedding.len() != self.weights.len() {
            return 0.0;
        }
        sigmoid(self.bias + SCALE * dot(&self.weights, embedding))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vectors near one of two directions, a little noisy.
    fn cluster(dir: usize, n: usize, seed: u32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v: Vec<f32> = (0..16)
                    .map(|j| (((i as u32 * 31 + j as u32 * 17 + seed) % 7) as f32 - 3.0) * 0.02)
                    .collect();
                v[dir] += 1.0;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn learns_which_side_is_which() {
        let mut examples: Vec<(Vec<f32>, bool)> =
            cluster(0, 6, 1).into_iter().map(|e| (e, true)).collect();
        examples.extend(cluster(1, 40, 2).into_iter().map(|e| (e, false)));
        let probe = Probe::train(&examples).expect("enough labels");
        for e in cluster(0, 5, 9) {
            assert!(probe.probability(&e) > 0.8);
        }
        for e in cluster(1, 5, 9) {
            assert!(probe.probability(&e) < 0.2);
        }
    }

    #[test]
    fn waits_for_enough_of_each_label() {
        let examples: Vec<(Vec<f32>, bool)> = cluster(0, 4, 1)
            .into_iter()
            .map(|e| (e, true))
            .chain(cluster(1, 50, 2).into_iter().map(|e| (e, false)))
            .collect();
        assert!(Probe::train(&examples).is_none());
    }

    #[test]
    fn a_vector_of_the_wrong_size_is_not_hidden() {
        let examples: Vec<(Vec<f32>, bool)> = cluster(0, 6, 1)
            .into_iter()
            .map(|e| (e, true))
            .chain(cluster(1, 6, 2).into_iter().map(|e| (e, false)))
            .collect();
        let probe = Probe::train(&examples).expect("enough labels");
        assert_eq!(probe.probability(&[1.0, 2.0]), 0.0);
    }
}
