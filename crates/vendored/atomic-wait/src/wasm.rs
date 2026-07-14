//! wasm32-wasi futex emulation over `std::sync::Condvar`.
//!
//! The real wasm futex instructions (`memory.atomic.wait32`/`notify`) are
//! unstable library features on stable rustc (`stdarch_wasm_atomic_wait`),
//! so this backend parks on a fixed table of address-hashed condvar
//! buckets instead — std itself is precompiled with thread support on
//! wasm32-wasip1-threads. Bucket collisions and `notify_all` on `wake_one`
//! only produce spurious returns, which this crate's contract permits.

extern crate std;

use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};

struct Bucket {
    lock: Mutex<()>,
    cond: Condvar,
}

const BUCKETS: usize = 64;
static TABLE: [Bucket; BUCKETS] = [const {
    Bucket {
        lock: Mutex::new(()),
        cond: Condvar::new(),
    }
}; BUCKETS];

fn bucket(atomic: *const AtomicU32) -> &'static Bucket {
    &TABLE[(atomic as usize >> 2) % BUCKETS]
}

pub fn wait(atomic: &AtomicU32, value: u32) {
    let bucket = bucket(atomic);
    let guard = bucket.lock.lock().unwrap();
    // Checked under the bucket lock: a waker stores the new value and then
    // takes this lock before notifying, so it cannot slip into the gap
    // between this check and the wait.
    if atomic.load(Ordering::SeqCst) != value {
        return;
    }
    // A single wait, no re-check loop: returning spuriously is allowed.
    drop(bucket.cond.wait(guard).unwrap());
}

fn wake(atomic: *const AtomicU32) {
    let bucket = bucket(atomic);
    drop(bucket.lock.lock().unwrap());
    // notify_all even for wake_one: waiters on colliding addresses may
    // absorb a notify_one meant for another address, losing the wakeup.
    bucket.cond.notify_all();
}

pub fn wake_one(atomic: *const AtomicU32) {
    wake(atomic);
}

pub fn wake_all(atomic: *const AtomicU32) {
    wake(atomic);
}
