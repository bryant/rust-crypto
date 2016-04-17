// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Machinery to compute the [Argon2](https://github.com/P-H-C/phc-winner-/blob/master/-specs.pdf) 
//! password hashing algorithm. 
//! 
//! The `argon2` module computes the [Argon2](https://github.com/P-H-C/phc-winner-/blob/master/-specs.pdf) 
//! password hashing algorithm through a simple method, `crypto::argon2::argon2()`. 
//!
//! This module implements Argon2, it was written by @bryant originally in [argon2rs](https://github.com/bryant/argon2rs). 
//! The API for this module follows a similar pattern to [`crypto::scrypt`](../scrypt/index.html). 
//!
//! # Provided quick access methods
//! 
//! * [`a2hash`](./fn.a2hash.html) -- Takes a password, salt, and `Algorithm` object and produces a hash.
//! * [`simple2d`](./fn.simple2d.html) -- Accepts a password, salt, variant and uses default parameters.
//! * [`simple2i`](./fn.simple2i.html) -- Accepts a password and salt, uses default parameters.
//! 
//! ## Examples 
//! 
//! Simple Argon2d hash: 
//!
//! Simple 16-byte Argon2d hash with 8 lanes (threads) and 1MB of memory:
//! 
//! ``` 
//! # extern crate rustc_serialize;
//! # extern crate crypto; // this is infuriating
//! # fn main() {
//! use crypto::argon2::{ a2hash, Algorithm };
//! use rustc_serialize::hex::ToHex; // for nice hashes
//!
//! // Expected hash result
//! const HASH: &'static str = "a36e28d37464eadfafd505205b3f3100";
//! 
//! // Set 8 lanes, and 1MB of memory
//! let a2 = Algorithm::argon2d()
//!                 .hash_length(16)        // Set the length smaller for docs :)
//!                 .lanes(8)               // 8 lanes 
//!                 .memory_size(1 * 1024); // 1 * 1024 == 1MB
//! 
//! // Run Argon2 with password: "football" and salt: "super salt"
//! let hash = a2hash("football", "super salt", &a2).unwrap();
//! 
//! // Make sure they're equal
//! assert_eq!(hash.to_hex(), HASH);
//! # }
//! ```

// ********************************************************************************************************************
// ** This test case is rough on a lot of computers, it should be verified regularly, but it's pretty slow. Thus it 
// ** is currenly ignored. Debug mode is the main culprit for this.
// ********************************************************************************************************************

//!
//! Large memory-hash using Argon2i with 12 lanes, 12 block passes and 1GB of memory
//!
//! ```ignore
//! // WARNING: This is very slow on older or busy computers. It is also slow in debug mode.
//! # extern crate rustc_serialize;
//! # extern crate crypto; // this is infuriating
//! # fn main() {
//! use rustc_serialize::hex::ToHex; // for nice hashes
//! 
//! use crypto::argon2::{ Algorithm, a2hash };
//!
//! // Expected hash
//! const HASH: &'static str = concat!("024d11660b08a1c38dbb2c048f48cbda",
//!                                    "c76a22547367a0f17fc32a63232c6dd7");
//! 
//! // Set 12 lanes, 5 passes, and 1GB memory
//! let params = Algorithm::argon2d()
//!                 .hash_length(32)                 // 32 bytes of hash
//!                 .lanes(12)                       // 12 threads  
//!                 .passes(5)                       // 5 passes
//!                 .memory_size(1 * 1024 * 1024);   // 1 * 1024 * 1024 == 1GB
//! 
//! // Run Argon2 with password: "master" and salt: "super salt"
//! let hash = a2hash("master", "super salt", &params).unwrap();
//! 
//! // Make sure they're equal
//! assert_eq!(hash.to_hex(), HASH);
//! # }
//! ```
//! 
//! Utilizing *all* of the parameters!
//!
//! ```
//! # extern crate rustc_serialize;
//! # extern crate crypto; // this is infuriating
//! # fn main() {
//! use rustc_serialize::hex::ToHex; // for nice hashes
//! 
//! use crypto::argon2::{ Algorithm, a2hash };
//!
//! // Expected hash
//! const HASH: &'static str = concat!("10c500f940373f979a53832d0726cce4",
//!                                    "c0817584fdb4d7ede19ee6be77008773");
//! 
//! // Set 4 lanes, 4 passes, and 350MB memory
//! let alg = Algorithm::argon2d()
//!                 .hash_length(32)                // 32 bytes of hash
//!                 .lanes(4)                       // 4 threads  
//!                 .passes(4)                      // 4 passes
//!                 .memory_size(350 * 1024);       // 350 * 1024 == 350MB
//! 
//! let mut buff = vec![0u8; 32]; // 32 bytes
//! alg.build().unwrap()
//!     .password("welcome")                    // Password
//!     .salt("More Salt!")                     // Salt
//!     .assoc_data(&[34, 32, 19, 09, 0xff])    // Associated data, `x`
//!     .secret(&[03, 43, 32])                  // Secret, `k`! don't tell anyone!
//!     .hash_inplace(buff.as_mut_slice())      // hash into the buffer we alloc'd before
//!     .unwrap();                              // Alternately, could have used .hash() which 
//!                                             //      returns a `Result<Vec<u8>, ParamErr>`
//! 
//! // Make sure they're equal
//! assert_eq!(buff.to_hex(), HASH);
//! # }
//! ```
//! 
//! # Builders: Saving State for repeated hashing patterns 
//! 
//! The `Algorithm` struct allows for consistent repeated hashes by maintaining a copy of the builder  
//! while performing multiple hashes. The instances are thread-safe.
//!
//! The `HashBldr` struct uses references to remove unecessary allocations, but does implement `Clone` if needed. 
//!
//! ## Reuse in Parallel Example
//! 
//! The following example shows reuse and checking in a multi-threaded environment.
//! 
//! ``` 
//! # extern crate rustc_serialize;
//! # extern crate crypto; // this is infuriating
//! # fn main() {
//! use rustc_serialize::hex::ToHex; // for nice hashes
//! 
//! use crypto::argon2::{ Algorithm };
//! use std::thread;
//! use std::sync::{ Arc, Mutex };
//! use std::time::Duration;
//! 
//! // Passwords!
//! const PASSWORDS: [&'static str; 3] = ["master", "princess", "12345678"];
//!
//! // Expected hashes!
//! const HASHES: [&'static str; 3] = [concat!("11b4c46b4fb6f00536c4a86b0b4c338a",
//!                                            "2ed30f32b35f8572f799df9721dae7cb"),
//!                                    concat!("265ca612224064703a0def52621b6a0c",
//!                                            "2729907f406af6b44ee1dc08c43f4c28"),
//!                                    concat!("e36a674caa556f0e70c9790d652cb6d7",
//!                                            "f8ce83916d74b1448194d36e44ea8fff")];
//! 
//! // Set 2 lanes, 3 passes, and 100MB memory per algorithm run
//! let alg_params = Arc::new(Algorithm::argon2d()
//!                             .hash_length(32)                // 32 bytes of hash
//!                             .lanes(2)                       // 2 threads  
//!                             .passes(3)                      // 3 passes
//!                             .memory_size(100 * 1024));      // 100 * 1024 == 100MB
//! let results = Arc::new(vec![Mutex::new(vec![0u8; 32]), 
//!                             Mutex::new(vec![0u8; 32]), 
//!                             Mutex::new(vec![0u8; 32])]);
//! 
//! // Compute all of the hashes in parallel
//! for i in 0..PASSWORDS.len() {
//!     let pass = PASSWORDS[i]; // access it this way to increase the scope of the value for the 
//!                              // thread closure later   
//!
//!     // clone the Arcs
//!     let alg = alg_params.clone();
//!     let results = results.clone();
//!     
//!     thread::spawn(move || {
//!         let mut result = results[i].lock().unwrap();
//!         alg.build().unwrap()
//!             .password(pass)
//!             .salt("super duper salt")
//!             .hash_inplace(result.as_mut_slice())
//!             .unwrap();
//!     });
//! }
//! 
//! // Go to sleep and wait for them to finish. If this was not enough time, 
//! // the `Mutex::lock()` will force us to wait later
//! thread::sleep(Duration::from_millis(50));
//! 
//! // Now check that the hashes are correct!
//! for (i, expected) in HASHES.iter().enumerate() {
//!     let result = results[i].lock().unwrap(); // lock the result arrays
//!     let hex = result.to_hex();
//! 
//!     println!("{}: left={} ;; right={}", i, hex, expected);
//!     assert_eq!(hex, *expected);
//! }
//! # }
//! ```
//!

use std::mem;
use std::fmt;

use std::iter::FromIterator;

use blake2b::Blake2b;

use digest::Digest;

use cryptoutil::copy_memory;

/// Denotes which Argon2 algorithm to use 
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum Variant {
    /// Argon2d is data dependent 
    Argon2d = 0,
    
    /// Argon2i is data independent
    Argon2i = 1,
}

const ARGON2_BLOCK_BYTES: usize = 1024;
const ARGON2_VERSION: u32 = 0x10;
const DEF_B2HASH_LEN: usize = 64;
const SLICES_PER_LANE: u32 = 4;
const DEF_HASH_LEN: usize = 64;

// from run.c
const T_COST_DEF: u32 = 3;
const LOG_M_COST_DEF: u32 = 12;
const LANES_DEF: u32 = 1;

macro_rules! per_block {
    (u8) => { ARGON2_BLOCK_BYTES };
    (u64) => { ARGON2_BLOCK_BYTES / 8 };
}

fn split_u64(n: u64) -> (u32, u32) {
    ((n & 0xffffffff) as u32, (n >> 32) as u32)
}

type Block = [u64; per_block!(u64)];

fn zero() -> Block { [0; per_block!(u64)] }

fn xor_all(blocks: &Vec<&Block>) -> Block {
    let mut rv: Block = zero();
    for (idx, d) in rv.iter_mut().enumerate() {
        *d = blocks.iter().fold(0, |n, &&blk| n ^ blk[idx]);
    }
    rv
}

fn as32le(k: u32) -> [u8; 4] { unsafe { mem::transmute(k.to_le()) } }

fn len32(t: &[u8]) -> [u8; 4] { as32le(t.len() as u32) }

fn as_u8_mut(b: &mut Block) -> &mut [u8] {
    let rv: &mut [u8; per_block!(u8)] = unsafe { mem::transmute(b) };
    rv
}

fn as_u8(b: &Block) -> &[u8] {
    let rv: &[u8; per_block!(u8)] = unsafe { mem::transmute(b) };
    rv
}

macro_rules! b2hash {
    ($($bytes: expr),*) => {
        {
            let mut out: [u8; DEF_B2HASH_LEN] = unsafe { mem::uninitialized() };
            b2hash!(&mut out; $($bytes),*);
            out
        }
    };
    ($out: expr; $($bytes: expr),*) => {
        {
            let mut b = Blake2b::new($out.len());
            $(b.input($bytes));*;
            b.result($out);
        }
    };
}

#[cfg_attr(rustfmt, rustfmt_skip)]
fn h0(lanes: u32, hash_length: u32, memory_kib: u32, passes: u32, version: u32,
      variant: Variant, p: &[u8], s: &[u8], k: &[u8], x: &[u8])
      -> [u8; 72] {
    let mut rv = [0 as u8; 72];
    b2hash!(&mut rv[0..DEF_B2HASH_LEN];
            &as32le(lanes), &as32le(hash_length), &as32le(memory_kib),
            &as32le(passes), &as32le(version), &as32le(variant as u32),
            &len32(p), p,
            &len32(s), s,
            &len32(k), k,
            &len32(x), x);
    rv
}


// Ideally, this doesn't really need to be `pub`, but for testing it helps a lot. 
pub struct Argon2 {
    blocks: Vec<Block>,
    
    /// Number of passes used in block matrix iterations. This forces the hash to take longer. 
    /// Must be larger than or equal to `1`.
    passes: u32,
    
    /// The number of lanes used, this increases the degree of parallelism when memory is filled during hash 
    /// computation. Setting this to `N` instructs argon2 to partition the block matrix into `N` lanes, 
    /// simultaneously filling the blocks. 
    lanes: u32,
    
    /// Truncated value of memory `(memory_kib / (4 * lanes)) * 4`
    lanelen: u32,
    
    /// The amount of memory used per block in KiB (1 KiB == 1024 Bytes). Increasing this forces the hash to use more
    /// memory to thwart ASIC attacks (for now). This value must be greater than or equal to `8 * lanes`.
    origkib: u32,
    
    /// The variant (flavour) of Argon2 to use, for password hashing, use `Variant::Argon2i`. 
    variant: Variant,
    
}

impl Argon2 {
    fn new(passes: u32, lanes: u32, memory_kib: u32, variant: Variant)
               -> Argon2 {
        assert!(lanes >= 1 && memory_kib >= 8 * lanes && passes >= 1);
        let lanelen = memory_kib / (4 * lanes) * 4;
        
        Argon2 {
            blocks: (0..lanelen * lanes).map(|_| zero()).collect(),
            passes: passes,
            lanelen: lanelen,
            lanes: lanes,
            origkib: memory_kib,
            variant: variant,
        }
    }

    fn hash(&mut self, out: &mut [u8], p: &[u8], s: &[u8], k: &[u8],
                x: &[u8]) {
        let h0 = self.h0(out.len() as u32, p, s, k, x);

        // TODO: parallelize
        for l in 0..self.lanes {
            self.fill_first_slice(h0, l);
        }

        // finish first pass. slices have to be filled in sync.
        for slice in 1..4 {
            for l in 0..self.lanes {
                self.fill_slice(0, l, slice, 0);
            }
        }

        for p in 1..self.passes {
            for s in 0..SLICES_PER_LANE {
                for l in 0..self.lanes {
                    self.fill_slice(p, l, s, 0);
                }
            }
        }

        let lastcol: Vec<&Block> = Vec::from_iter((0..self.lanes).map(|l| {
            &self.blocks[self.blkidx(l, self.lanelen - 1)]
        }));

        h_prime(out, as_u8(&xor_all(&lastcol)));
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn h0(&self, tau: u32, p: &[u8], s: &[u8], k: &[u8], x: &[u8]) -> [u8; 72] {
        h0(self.lanes, tau, self.origkib, self.passes, ARGON2_VERSION,
           self.variant, p, s, k, x)
    }

    fn blkidx(&self, row: u32, col: u32) -> usize {
        (self.lanelen * row + col) as usize
    }

    fn fill_first_slice(&mut self, mut h0: [u8; 72], lane: u32) {
        // fill the first (of four) slice
        copy_memory(&as32le(lane), &mut h0[68..72]);

        copy_memory(&as32le(0), &mut h0[64..68]);
        let zeroth = self.blkidx(lane, 0);
        h_prime(as_u8_mut(&mut self.blocks[zeroth]), &h0);

        copy_memory(&as32le(1), &mut h0[64..68]);
        let first = self.blkidx(lane, 1);
        h_prime(as_u8_mut(&mut self.blocks[first]), &h0);

        // finish rest of first slice
        self.fill_slice(0, lane, 0, 2);
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn fill_slice(&mut self, pass: u32, lane: u32, slice: u32, offset: u32) {
        let mut jgen = Gen2i::new(offset as usize, pass, lane, slice,
                                  self.blocks.len() as u32, self.passes);
        let slicelen = self.lanelen / SLICES_PER_LANE;

        for idx in offset..slicelen {
            let (j1, j2) = if self.variant == Variant::Argon2i {
                jgen.nextj()
            } else {
                let i = self.prev(self.blkidx(lane, slice * slicelen + idx));
                split_u64((self.blocks[i])[0])
            };
            self.fill_block(pass, lane, slice, idx, j1, j2);
        }
    }

    fn fill_block(&mut self, pass: u32, lane: u32, slice: u32, idx: u32,
                  j1: u32, j2: u32) {
        let slicelen = self.lanelen / SLICES_PER_LANE;
        let ls = self.lanes;
        let z = index_alpha(pass, lane, slice, ls, idx, slicelen, j1, j2);

        let zth = match (pass, slice) {
            (0, 0) => self.blkidx(lane, z),
            _ => self.blkidx(j2 % self.lanes, z),
        };

        let cur = self.blkidx(lane, slice * slicelen + idx);
        let pre = self.prev(cur);
        let (wr, rd, refblk) = get3(&mut self.blocks, cur, pre, zth);
        g(wr, rd, refblk);
    }

    fn prev(&self, block_index: usize) -> usize {
        match block_index % self.lanelen as usize {
            0 => block_index + self.lanelen as usize - 1,
            _ => block_index - 1,       
        }
    }
}

fn get3<T>(vector: &mut Vec<T>, wr: usize, rd0: usize, rd1: usize)
           -> (&mut T, &T, &T) {
    assert!(wr != rd0 && wr != rd1 && wr < vector.len() &&
            rd0 < vector.len() && rd1 < vector.len());
    let p: *mut [T] = &mut vector[..];
    let rv = unsafe { (&mut (*p)[wr], &(*p)[rd0], &(*p)[rd1]) };
    rv
}

fn h_prime(out: &mut [u8], input: &[u8]) {
    if out.len() <= DEF_B2HASH_LEN {
        b2hash!(out; &len32(out), input);
    } else {
        let mut tmp = b2hash!(&len32(out), input);
        copy_memory(&tmp, &mut out[0..DEF_B2HASH_LEN]);
        let mut wr_at: usize = 32;

        while out.len() - wr_at > DEF_B2HASH_LEN {
            b2hash!(&mut tmp; &tmp);
            copy_memory(&tmp, &mut out[wr_at..wr_at + DEF_B2HASH_LEN]);
            wr_at += DEF_B2HASH_LEN / 2;
        }

        let len = out.len() - wr_at;
        b2hash!(&mut out[wr_at..wr_at + len]; &tmp);
    }
}

// from opt.c
fn index_alpha(pass: u32, lane: u32, slice: u32, lanes: u32, sliceidx: u32,
               slicelen: u32, j1: u32, j2: u32)
               -> u32 {
    let lanelen = slicelen * 4;
    let r: u32 = match (pass, slice, j2 % lanes == lane) {
        (0, 0, _) => sliceidx - 1,
        (0, _, false) => slice * slicelen - if sliceidx == 0 { 1 } else { 0 },
        (0, _, true) => slice * slicelen + sliceidx - 1,
        (_, _, false) => lanelen - slicelen - if sliceidx == 0 { 1 } else { 0 },
        (_, _, true) => lanelen - slicelen + sliceidx - 1,
    };

    let (r_, j1_) = (r as u64, j1 as u64);
    let relpos: u32 = (r_ - 1 - (r_ * (j1_ * j1_ >> 32) >> 32)) as u32;

    match (pass, slice) {
        (0, _) | (_, 3) => relpos % lanelen,
        _ => (slicelen * (slice + 1) + relpos) % lanelen,
    }
}

struct Gen2i {
    arg: Block,
    pseudos: Block,
    idx: usize,
}

impl Gen2i {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn new(start_at: usize, pass: u32, lane: u32, slice: u32, totblocks: u32,
           totpasses: u32)
           -> Gen2i {
        let mut rv = Gen2i { arg: zero(), pseudos: zero(), idx: start_at };
        let args = [pass, lane, slice, totblocks, totpasses,
                    Variant::Argon2i as u32];
        for (k, v) in rv.arg.iter_mut().zip(args.into_iter()) {
            *k = *v as u64;
        }
        rv.more();
        rv
    }

    fn more(&mut self) {
        self.arg[6] += 1;
        g_two(&mut self.pseudos, &self.arg);
    }

    fn nextj(&mut self) -> (u32, u32) {
        let rv = split_u64(self.pseudos[self.idx]);
        self.idx = (self.idx + 1) % per_block!(u64);
        if self.idx == 0 {
            self.more();
        }
        rv
    }
}

// g x y = let r = x `xor` y in p_col (p_row r) `xor` r,
// very simd-able.
fn g(dest: &mut Block, lhs: &Block, rhs: &Block) {
    for (d, (l, r)) in dest.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
        *d = *l ^ *r;
    }

    for row in 0..8 {
        p_row(row, dest);
    }
    // column-wise, 2x u64 groups
    for col in 0..8 {
        p_col(col, dest);
    }

    for (d, (l, r)) in dest.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
        *d = *d ^ *l ^ *r;
    }
}

// g2 y = g 0 (g 0 y). used for data-independent index generation.
fn g_two(dest: &mut Block, src: &Block) {
    *dest = *src;

    for row in 0..8 {
        p_row(row, dest);
    }
    for col in 0..8 {
        p_col(col, dest);
    }

    for (d, s) in dest.iter_mut().zip(src.iter()) {
        *d = *d ^ *s;
    }

    let tmp: Block = *dest;

    for row in 0..8 {
        p_row(row, dest);
    }
    for col in 0..8 {
        p_col(col, dest);
    }

    for (d, s) in dest.iter_mut().zip(tmp.iter()) {
        *d = *d ^ *s;
    }
}

macro_rules! p {
    ($v0: expr, $v1: expr, $v2: expr, $v3: expr,
     $v4: expr, $v5: expr, $v6: expr, $v7: expr,
     $v8: expr, $v9: expr, $v10: expr, $v11: expr,
     $v12: expr, $v13: expr, $v14: expr, $v15: expr) => {
        g_blake2b!($v0, $v4, $v8, $v12); g_blake2b!($v1, $v5, $v9, $v13);
        g_blake2b!($v2, $v6, $v10, $v14); g_blake2b!($v3, $v7, $v11, $v15);
        g_blake2b!($v0, $v5, $v10, $v15); g_blake2b!($v1, $v6, $v11, $v12);
        g_blake2b!($v2, $v7, $v8, $v13); g_blake2b!($v3, $v4, $v9, $v14);
    };
}

macro_rules! g_blake2b {
    ($a: expr, $b: expr, $c: expr, $d: expr) => {
        $a = $a.wrapping_add($b).wrapping_add(lower_mult($a, $b));
        $d = ($d ^ $a).rotate_right(32);
        $c = $c.wrapping_add($d).wrapping_add(lower_mult($c, $d));
        $b = ($b ^ $c).rotate_right(24);
        $a = $a.wrapping_add($b).wrapping_add(lower_mult($a, $b));
        $d = ($d ^ $a).rotate_right(16);
        $c = $c.wrapping_add($d).wrapping_add(lower_mult($c, $d));
        $b = ($b ^ $c).rotate_right(63);

    }
}

fn p_row(row: usize, b: &mut Block) {
    p!(b[16 * row + 0],
       b[16 * row + 1],
       b[16 * row + 2],
       b[16 * row + 3],
       b[16 * row + 4],
       b[16 * row + 5],
       b[16 * row + 6],
       b[16 * row + 7],
       b[16 * row + 8],
       b[16 * row + 9],
       b[16 * row + 10],
       b[16 * row + 11],
       b[16 * row + 12],
       b[16 * row + 13],
       b[16 * row + 14],
       b[16 * row + 15]);
}

fn p_col(col: usize, b: &mut Block) {
    p!(b[2 * col + 16 * 0],
       b[2 * col + 16 * 0 + 1],
       b[2 * col + 16 * 1],
       b[2 * col + 16 * 1 + 1],
       b[2 * col + 16 * 2],
       b[2 * col + 16 * 2 + 1],
       b[2 * col + 16 * 3],
       b[2 * col + 16 * 3 + 1],
       b[2 * col + 16 * 4],
       b[2 * col + 16 * 4 + 1],
       b[2 * col + 16 * 5],
       b[2 * col + 16 * 5 + 1],
       b[2 * col + 16 * 6],
       b[2 * col + 16 * 6 + 1],
       b[2 * col + 16 * 7],
       b[2 * col + 16 * 7 + 1]);
}

fn lower_mult(a: u64, b: u64) -> u64 {
    fn lower32(k: u64) -> u64 { k & 0xffffffff }
    lower32(a).wrapping_mul(lower32(b)).wrapping_mul(2)
}

/// Builder which creates `Hash` instances for reuse.  
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Algorithm {
    /// Number of passes used in block matrix iterations. This forces the hash to take longer. 
    /// Must be larger than or equal to `1`.
    passes: u32,
    
    /// How large of a hash to create in bytes, defaults to 32.
    hash_length: usize,
    
    /// The number of lanes used, this increases the degree of parallelism when memory is filled during hash 
    /// computation. Setting this to `N` instructs argon2 to partition the block matrix into `N` lanes, 
    /// simultaneously filling the blocks. 
    lanes: u32,
    
    /// The amount of memory used per block in KiB (1 KiB == 1024 Bytes). Increasing this forces the hash to use more
    /// memory to thwart ASIC attacks (for now). This value must be greater than or equal to `8 * lanes`.
    memory_size: u32,
    
    /// The variant (flavour) of Argon2 to use, for password hashing, use `Variant::Argon2i`. 
    variant: Variant,
}

/// Build up a single hash (reusable)
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct HashBldr {
    
    /// Reference to the parent `Algorithm` instance that created it 
    alg_bldr: Algorithm, 
    
    /// Extra data used in computing, this is also known as `x`
    assoc_data: Vec<u8>,
    
    /// Secret data, also known as `k`
    secret: Vec<u8>,
    
    /// Password to hash 
    password: Vec<u8>,
    
    /// Salt to stretch the `password`
    salt: Vec<u8>,
}

/// Error value used to describe errors in Parameters
#[derive(Eq, PartialEq, Clone, Debug)]
pub enum ParamErr {
    /// Zero passes were specified
    ZeroPassesSpecified, 
    
    /// A Lanes count of Zero was specified
    ZeroLanesSpecified, 
    
    /// Too little memory was requested for the total operation, it must be a **minimum** of `8 * lanes`. 
    TooLittleMemoryForLanes { 
        /// Number of lanes requested 
        lanes: u32, 
        /// Amount of memory in KiB that was requested
        memory: u32 
    },
    
    /// The secret value passed is too large
    SecretTooLarge(usize),
    
    /// The length of the value of associated data is too large for the algorithm, it must be between 
    /// `0` -> `2^32 - 1` Bytes.
    AssociatedDataTooLarge(usize),
    
    /// Password is 0 length which is too short. 
    PasswordEmpty,
    
    /// Salt length is invalid, it needs to be in `[8, 2**32 - 1]` 
    InvalidSaltLength(usize),
    
    /// Hash length is invalid, it needs to be between `[4, 2**32 - 1]`
    InvalidHashLength(usize),
    
    /// A buffer was passed for inplace hashing, however, the buffer size does not match 
    InvalidHashBufferLength { 
        /// Buffer length
        buffer_len: usize,
        
        /// The set length
        hash_length: usize
    }
}

impl Algorithm {

    /// Load a new `Builder` instance with default values for `Argon2i`.
    /// Use this version for password hashing. 
    pub fn argon2i() -> Algorithm {
        Algorithm {
            passes: T_COST_DEF, 
            lanes: LANES_DEF,
            memory_size: 1 << LOG_M_COST_DEF, // bad name..
            variant: Variant::Argon2i,
            
            hash_length: DEF_HASH_LEN,
        }
    }
    
    /// Load a new `Builder` instance with default values for `Argon2d`
    pub fn argon2d() -> Algorithm {
        Algorithm {
            passes: T_COST_DEF, 
            lanes: LANES_DEF,
            memory_size: 1 << LOG_M_COST_DEF, // bad name..
            variant: Variant::Argon2d,
            
            hash_length: DEF_HASH_LEN,
        }
    }
    
    /// Set the number of parallel lanes used 
    /// Set the number of parallel lanes used when computing the initial block. This has a default constructed value of 
    /// `1` from `argon2::defaults::LANES`. This value must be greater than 0. A suggestion is to use `num_cpus::get()`
    /// from the [num_cpus](https://github.com/seanmonstar/num_cpus). 
    /// 
    /// # Examples
    /// 
    /// ```
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2::{ a2hash, Algorithm };
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    ///
    /// const FIFTY_MB: u32 = 50 * 1024; 
    /// // Change the size to be 50MB and 2 parallel lanes 
    /// let ab = Algorithm::argon2i()
    ///             .hash_length(8)
    ///             .passes(1)                  // This just slows it down
    ///             .lanes(2)                   // 4 lanes
    ///             .memory_size(FIFTY_MB);     // 50MB
    ///
    /// let hash = a2hash("master", "saltsalt", &ab).unwrap();
    /// assert_eq!("f52a5f0f72c87029", hash.to_hex());
    /// # }
    /// ``` 
    pub fn lanes(&self, lanes: u32) -> Algorithm {
        let mut ab = self.clone();
        ab.lanes = lanes;
        
        ab
    }
    
    /// Set the total size of the blocks used in the hash matrix.
    /// Set the total size of blocks used in memory allocation. This has a default constructed value of `4096` KiB from 
    /// `argon2::defaults::KIB`. This value must be `>= 8 * lanes`. This will panic at `#build()` if the value is less 
    /// than `8 * lanes`, and at method call if `< 8`. 
    /// 
    /// # Examples
    /// 
    /// ```
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2::{ a2hash, Algorithm };
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    ///
    /// // Change the size to be 150MB
    /// const ONE_FIFTY_MB: u32 = 150 * 1024; 
    /// let ab = Algorithm::argon2i()
    ///             .passes(1)
    ///             .lanes(2)
    ///             .hash_length(8)
    ///             .memory_size(ONE_FIFTY_MB); // 150MB
    ///
    /// let hash = a2hash("princess", "saltsalt", &ab).unwrap();
    /// assert_eq!("c82552f5c18f4c98", hash.to_hex());
    /// # }
    /// ``` 
    pub fn memory_size(&self, size: u32) -> Algorithm {
        let mut ab = self.clone();
        ab.memory_size = size;
        
        ab
    }
    
    /// Set the number of rotation passes per block.
    /// 
    /// Set the number of passes used when rotating blocks. This has a default constructed value of `3` from 
    /// `T_COST_DEF`. This value must be greater than 0.  
    /// 
    /// # Examples
    /// 
    /// ```
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2::{ a2hash, Algorithm };
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    ///
    /// let ab = Algorithm::argon2i().hash_length(16).passes(2);
    /// let hash = a2hash("qwerty", "saltsalt", &ab).unwrap();
    /// assert_eq!("1fa37316c046c4dde19a3e110c97473d", hash.to_hex());
    /// # }
    /// ``` 
    pub fn passes(&self, passes: u32) -> Algorithm {
        let mut ab = self.clone();
        ab.passes = passes;
        
        ab
    }
    
    /// Set the length of the resulting hash.
    /// 
    /// # Example
    /// 
    /// ```
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2::{ a2hash, Algorithm };
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    ///
    /// let ab = Algorithm::argon2i().hash_length(8);
    /// let hash = a2hash("password", "saltsalt", &ab).unwrap();
    /// assert_eq!(8, hash.len());
    /// assert_eq!("de1477d409757354", hash.to_hex());
    /// # }
    /// ``` 
    pub fn hash_length(&self, length: usize) -> Algorithm {
        let mut ab = self.clone();
        
        ab.hash_length = length;
       
        ab
    }

    
    /// Change the Variant of Argon2 used 
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2::{ a2hash, Variant, Algorithm };
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    ///
    /// let ab = Algorithm::argon2i().hash_length(8);
    /// let hash_i = a2hash("password", "saltsalt", &ab).unwrap();
    /// assert_eq!("de1477d409757354", hash_i.to_hex());
    /// 
    /// // Now use Argon2d
    /// let hash_d = a2hash("password", "saltsalt", &ab.variant(Variant::Argon2d)).unwrap();
    /// assert_eq!("8796b62914f50725", hash_d.to_hex());
    /// 
    /// // The different variants create different hashes. 
    /// assert!(hash_d != hash_i, "Hashes are equal! :(");
    /// # }
    /// ``` 
    pub fn variant(&self, variant: Variant) -> Algorithm {
        let mut ab = self.clone();
        
        ab.variant = variant;
        
        ab
    }
    
    /// Verify that the result is valid
    fn verify(&self) -> Result<(), ParamErr> {
         if self.passes == 0 {
            Result::Err(ParamErr::ZeroPassesSpecified)
        } else if self.lanes == 0 {
            Result::Err(ParamErr::ZeroLanesSpecified)
        } else if (self.lanes * 8) > self.memory_size {
            Result::Err(ParamErr::TooLittleMemoryForLanes {
                lanes: self.lanes,
                memory: self.memory_size,
            })
        } else if self.hash_length < 4 || self.hash_length > 0xFFFFFFFF { 
            Result::Err(ParamErr::InvalidHashLength(self.hash_length))  
        } else {
            Result::Ok(())
        }
    }

    /// Create a new `HashBldr` instance
    pub fn build(&self) -> Result<HashBldr, ParamErr> {
        self.verify().and_then(|_| {
             Result::Ok(HashBldr::new(self.clone()))
        })
    }
}

impl HashBldr {
    
    /// Create a new instance of the HashBldr, everything defaults to empty vec
    fn new(ab: Algorithm) -> HashBldr {
        HashBldr {
            alg_bldr: ab,
            
            assoc_data: vec![0u8; 0], 
            secret: vec![0u8; 0],
            
            password: vec![0u8; 0],
            salt: vec![0u8; 0],
        }
    }
    
    /// Set the associated data for a `HashBldr` instance, this is the variable `x` in the Argon2 algorithms.
    /// 
    /// # Example
    /// 
    /// ```
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2::{ Algorithm };
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    /// 
    /// const ASSOC_DATA: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
    ///
    /// let mut ab = Algorithm::argon2i()
    ///                 .hash_length(8)
    ///                 .build().unwrap();
    /// let hash = ab.assoc_data(&ASSOC_DATA)
    ///                 .password("qwerty")
    ///                 .salt("super salt")
    ///                 .hash().unwrap();
    /// assert_eq!(hash.to_hex(), "b03edb6399b78656");
    /// # }
    /// ```
    pub fn assoc_data(&mut self, x: &[u8]) -> &mut HashBldr {
        self.assoc_data.clear();
        self.assoc_data.extend_from_slice(x);
        
        self
    }
    
    /// Set the salt for a hash.
    /// 
    /// # Example
    /// 
    /// ``` 
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2;
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    ///
    /// const SALT: &'static str = "deadbeef";
    ///
    /// let mut ab = argon2::Algorithm::argon2i().hash_length(8).build().unwrap();
    /// let hash = ab.password("qwerty")
    ///                 .salt(&SALT)
    ///                 .hash().unwrap();
    /// assert_eq!(hash.to_hex(), "4c233cd0d5f78c98");
    /// # }
    /// ```
    pub fn salt(&mut self, salt: &str) -> &mut HashBldr {
        self.salt.clear();
        self.salt.extend_from_slice(salt.as_bytes());
       
        self
    }
    
    /// Set the secret data for a hash, this is `k` in the Argon2 algorithm.
    /// 
    /// # Example
    /// 
    /// ``` 
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2;
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    /// 
    /// const SECRET: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
    ///
    /// let mut ab = argon2::Algorithm::argon2i().hash_length(8).build().unwrap();
    /// let hash = ab.secret(&SECRET)
    ///                 .password("princess")
    ///                 .salt("super salt")
    ///                 .hash().unwrap();
    /// assert_eq!(hash.to_hex(), "5d6a66408ace4d92");
    /// # }
    /// ```
    pub fn secret(&mut self, k: &[u8]) -> &mut HashBldr {
        self.secret.clear();
        self.secret.extend_from_slice(k);
       
        self
    }
    
    
    /// Set the password for a hash
    /// 
    /// # Example
    /// 
     /// ``` 
    /// # extern crate rustc_serialize;
    /// # extern crate crypto; // this is infuriating
    /// # fn main() {
    /// use crypto::argon2;
    /// use rustc_serialize::hex::ToHex; // for nice hashes
    /// 
    /// const PASSWORD: &'static str = "hunter1";
    ///
    /// let mut hb = argon2::Algorithm::argon2i().hash_length(16).build().unwrap();
    /// let hash = hb.password(PASSWORD)
    ///                     .salt("super salt")
    ///                     .hash().unwrap();
    /// assert_eq!(hash.to_hex(), "24c92fd56c1b8e6b72c379681a8a4df7");
    /// # }
    /// ```
    pub fn password(&mut self, password: &str) -> &mut HashBldr {
        self.password.clear();
        self.password.extend_from_slice(password.as_bytes());
       
        self
    }
    
    /// Verify that the set parameters are OK, if not return `Result::Err`.
    fn verify(&self) -> Result<Argon2, ParamErr> {
        if (self.salt.len() < 8) || (self.salt.len() > 0xFFFFFFFF) {
            Result::Err(ParamErr::InvalidSaltLength(self.salt.len()))
        } else if self.password.len() == 0 {
            Result::Err(ParamErr::PasswordEmpty)
        } else if self.secret.len() > 32 {
            Result::Err(ParamErr::SecretTooLarge(self.secret.len()))
        } else if self.assoc_data.len() > (2 << 32 - 1) {
            Result::Err(ParamErr::AssociatedDataTooLarge(self.assoc_data.len()))
        } else {
            Result::Ok(Argon2::new(self.alg_bldr.passes,      self.alg_bldr.lanes, 
                                   self.alg_bldr.memory_size, self.alg_bldr.variant))
        }
    }
    
    /// Performs a hash of the stored parameters. 
    pub fn hash(&self) -> Result<Vec<u8>, ParamErr> {
        self.verify().and_then(|mut ag2| {
            let mut out = vec![0u8; self.alg_bldr.hash_length];
            
            ag2.hash(out.as_mut_slice(), &self.password, &self.salt, &self.secret, &self.assoc_data);
            
            Result::Ok(out)
        })
    }
    
    /// Performs a hash of the stored parameters on an inplace buffer
    pub fn hash_inplace(&self, out: &mut [u8]) -> Result<(), ParamErr> {
        
        self.verify().and_then(|mut ag2| {
            
            if out.len() != self.alg_bldr.hash_length {
                Result::Err(ParamErr::InvalidHashBufferLength {
                    buffer_len: out.len(),
                    hash_length: self.alg_bldr.hash_length
                })
            } else {
                // do the actual hash :)
                ag2.hash(out, &self.password, &self.salt, &self.secret, &self.assoc_data);
                
                Result::Ok(())
            }
        })
    }
}

// Pretty print the Variant
impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Variant::Argon2d => write!(f, "Argon2d"),
            Variant::Argon2i => write!(f, "Argon2i"),
        }
    }
}

// Pretty print the HashBldr
impl fmt::Display for HashBldr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // print a slice
        fn print_hex_slice(f: &mut fmt::Formatter, vc: &[u8]) -> fmt::Result {
            try!(write!(f, "["));
            for i in 0..(vc.len() - 1) {
                try!(write!(f, "{:02x}", vc[i]));
            }
            if let Some(v) = vc.get(vc.len() - 1) {
                try!(write!(f, "{:02x}", v));
            }
            write!(f, "]")
        }
        
        try!(write!(f, "{{alg: {}, assoc_data: ", self.alg_bldr));
        try!(print_hex_slice(f, &self.assoc_data));
        try!(write!(f, ", salt: "));
        try!(print_hex_slice(f, &self.salt));
        write!(f, "secret_set: {}, password_set: {}}}", self.secret.len() > 0, self.password.len() > 0)
    }
}

// Pretty print the builder
impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{passes: {}, hash_length: {} B, lanes: {}, memory: {} KiB, variant: {}}}",
            self.passes, self.hash_length, self.lanes, self.memory_size, self.variant)
    }
}

/// Compute the [Argon2](https://github.com/P-H-C/phc-winner-/blob/master/-specs.pdf) hash of a password and salt. 
/// This is a short-hand method to using the builders. 
/// 
/// # Arguments 
///
/// * `password` - The password to hash 
/// * `salt` - Salt used with the password 
/// * `params` - Parameters used to configure the Argon2 algorithm. 
/// 
/// # Example 
///
/// ```
/// # extern crate rustc_serialize;
/// # extern crate crypto; // this is infuriating
/// # fn main() {
///
/// use crypto::argon2::{ a2hash, Algorithm };
/// 
/// use rustc_serialize::hex::ToHex; // for simple serialization
///
/// // Compute a 16-byte Argon2d hash of "welcome" with "super salt" running in 12 lanes 
/// // with 5MB of memory and 2 passes per iteration. 
/// let hash = a2hash("welcome", "super salt", &Algorithm::argon2d()
///                                                 .hash_length(16)
///                                                 .lanes(12)
///                                                 .memory_size(5 * 1024)
///                                                 .passes(2)).unwrap();
/// assert_eq!("8e5c596f755d5c941bd7256ac6876c10", hash.to_hex());
/// # }
/// ```
/// 
/// 
pub fn a2hash(password: &str, salt: &str, params: &Algorithm) -> Result<Vec<u8>, ParamErr> { 
    let mut hb = try!(params.build());
    
    hb.password(password)
        .salt(salt)
        .hash()
}


/// Computes a hash using `Argon2i` with a password and salt, and default parameters.
///
/// Computes a hash, storing it into `out`. This uses reasonable default parameters.
///
/// # Arguments 
/// 
/// * `password` - Password to hash 
/// * `salt` - Salt used with the password between `[8, 2 ** 32 - 1]` bytes 
/// 
/// # Examples 
/// 
/// ```
/// # extern crate rustc_serialize;
/// # extern crate crypto; // this is infuriating
/// # fn main() {
///
/// use crypto::argon2::simple2i;
/// 
/// use rustc_serialize::hex::ToHex; // for simple serialization
/// 
/// const EXPECTED: &'static str = concat!("d92f206d6220bd3f491809fb1ad9e54c9be2f13545b3fd3a9cfc7fbcc5f596dd", 
///                                        "b744ff23406d09c8c1a30cb6a1c5552a8197f0d15d93c8acceda0cfe46bb66a9");
///
/// // Compute a 64-byte Argon2i hash of "princess" with "super salt"
/// let hash = simple2i("princess", "super salt");
/// assert_eq!(hash.to_hex(), EXPECTED);
/// # }
/// ```
/// 
/// A thorough comparison between the `simple` and full version:
///
/// ```
/// # extern crate rustc_serialize;
/// # extern crate crypto; // this is infuriating
/// # fn main() {
/// use rustc_serialize::hex::ToHex; // for simple serialization
///
/// use crypto::argon2::{ Algorithm, simple2i };
///
/// // Compute a 64-byte Argon2i hash of "princess" with "super salt"
/// const PASSWORD: &'static str = "princess";
/// const SALT: &'static str = "super salt";
/// 
/// // Short version
/// let short = simple2i(PASSWORD, SALT);
/// 
/// // Long version with a lot more configuration available
/// let long = Algorithm::argon2i()
///                 .build()
///                 .and_then(|mut hb| {
///                     hb.salt(SALT)
///                         .password(PASSWORD)
///                         .hash()   
///                  }).unwrap(); // should normally check!
///
/// assert_eq!(&long.to_hex(), &short.to_hex());
/// # }
/// ```
pub fn simple2i(password: &str, salt: &str) -> [u8; DEF_HASH_LEN] {
    let mut out = [0; DEF_HASH_LEN];
    
    Algorithm::argon2i().build().and_then(|mut hb| {
        hb.password(password)
           .salt(salt)
           .hash_inplace(&mut out)
    }).unwrap(); // this won't fail.. in theory
    
    out
}

/// Computes a hash using `Argon2d` with a password and salt, and default parameters.
///
/// Computes a hash, storing it into `out`. This uses reasonable default parameters.
///
/// # Arguments 
/// 
/// * `password` - Password str to hash 
/// * `salt` - Salt used with the password between `[8, 2 ** 32 - 1]` bytes 
/// 
/// # Examples
/// 
/// ```
/// # extern crate rustc_serialize;
/// # extern crate crypto; // this is infuriating
/// # fn main() {
/// use rustc_serialize::hex::ToHex; // for simple serialization
///
/// use crypto::argon2::simple2d;
/// 
/// const EXPECTED: &'static str = concat!("4fdb203ebce7edaa5a44e887c301e14e12087b405aa847c7e03a8da0030bb2f6",
///                                        "f48a531b762561fd14bde03853525693c6cddedd042108327b7baedd127f30a6");
///
/// // Compute a 64-byte Argon2d hash of "princess" with "super salt"
/// let hash = simple2d("princess", "super salt");
/// assert_eq!(EXPECTED, hash.to_hex());
/// # }
/// ```
/// 
/// A thorough comparison between the `simple` and full version:
///
/// ```
/// # extern crate rustc_serialize;
/// # extern crate crypto; // this is infuriating
/// # fn main() {
/// use rustc_serialize::hex::ToHex; // for simple serialization
/// 
/// use crypto::argon2::{ Algorithm, simple2d };
///
/// // Compute a 64-byte Argon2d hash of "princess" with "super salt"
/// const PASSWORD: &'static str = "princess";
/// const SALT: &'static str = "super salt";
/// 
/// // Short version
/// let short = simple2d(PASSWORD, SALT);
/// 
/// // Long version with a lot more configuration available
/// let long = Algorithm::argon2d()
///                 .build()
///                 .and_then(|mut hb| {
///                     hb.salt(SALT)
///                         .password(PASSWORD)
///                         .hash()
///                  }).unwrap();
///
/// assert_eq!(&long.to_hex(), &short.to_hex());
/// # }
/// ```
pub fn simple2d(password: &str, salt: &str) -> [u8; DEF_HASH_LEN] {
    let mut out = [0; DEF_HASH_LEN];
    
    Algorithm::argon2d().build().and_then(|mut hb| {
        hb.password(password)
           .salt(salt)
           .hash_inplace(&mut out)
    }).unwrap(); // this won't fail.. in theory
    
    out
}
    
#[cfg(test)]
mod kat_tests {
    use std::fs::File;
    use std::iter::FromIterator;
    use std::io::Read;

    // from genkat.c
    const TEST_OUTLEN: usize = 32;
    const TEST_PWDLEN: usize = 32;
    const TEST_SALTLEN: usize = 16;
    const TEST_SECRETLEN: usize = 8;
    const TEST_ADLEN: usize = 12;

    fn u8info(prefix: &str, bytes: &[u8], print_length: bool) -> String {
        let bs = bytes.iter()
                      .fold(String::new(), |xs, b| xs + &format!("{:02x} ", b));
        let len = match print_length {
            false => ": ".to_string(),
            true => format!("[{}]: ", bytes.len()),
        };
        prefix.to_string() + &len + &bs

    }

    fn block_info(i: usize, b: &super::Block) -> String {
        b.iter().enumerate().fold(String::new(), |xs, (j, octword)| {
            xs + "Block " + &format!("{:004} ", i) + &format!("[{:>3}]: ", j) +
            &format!("{:0016x}", octword) + "\n"
        })
    }

    fn gen_kat(a: &mut super::Argon2, tau: u32, p: &[u8], s: &[u8], k: &[u8],
               x: &[u8])
               -> String {
        let eol = "\n";
        let mut rv = format!("======================================={:?}",
                             a.variant) + eol +
                     &format!("Memory: {} KiB, ", a.origkib) +
                     &format!("Iterations: {}, ", a.passes) +
                     &format!("Parallelism: {} lanes, ", a.lanes) +
                     &format!("Tag length: {} bytes", tau) +
                     eol + &u8info("Password", p, true) +
                     eol +
                     &u8info("Salt", s, true) +
                     eol + &u8info("Secret", k, true) +
                     eol +
                     &u8info("Associated data", x, true) +
                     eol;

        let h0 = a.h0(tau, p, s, k, x);
        rv = rv +
             &u8info("Pre-hashing digest",
                     &h0[..super::DEF_B2HASH_LEN],
                     false) + eol;

        // first pass
        for l in 0..a.lanes {
            a.fill_first_slice(h0, l);
        }
        for slice in 1..4 {
            for l in 0..a.lanes {
                a.fill_slice(0, l, slice, 0);
            }
        }

        rv = rv + eol + " After pass 0:" + eol;
        for (i, block) in a.blocks.iter().enumerate() {
            rv = rv + &block_info(i, block);
        }

        for p in 1..a.passes {
            for s in 0..super::SLICES_PER_LANE {
                for l in 0..a.lanes {
                    a.fill_slice(p, l, s, 0);
                }
            }

            rv = rv + eol + &format!(" After pass {}:", p) + eol;
            for (i, block) in a.blocks.iter().enumerate() {
                rv = rv + &block_info(i, block);
            }
        }

        let lastcol: Vec<&super::Block> =
            Vec::from_iter((0..a.lanes)
                               .map(|l| &a.blocks[a.blkidx(l, a.lanelen - 1)]));

        let mut out = vec![0; tau as usize];
        super::h_prime(&mut out, super::as_u8(&super::xor_all(&lastcol)));
        rv + &u8info("Tag", &out, false)
    }

    fn compare_kats(fexpected: &str, variant: super::Variant) {
        let mut f = File::open(fexpected).unwrap();
        let mut expected = String::new();
        f.read_to_string(&mut expected).unwrap();

        let mut a = super::Argon2::new(3, 4, 32, variant);
        let actual = gen_kat(&mut a,
                             TEST_OUTLEN as u32,
                             &[1; TEST_PWDLEN],
                             &[2; TEST_SALTLEN],
                             &[3; TEST_SECRETLEN],
                             &[4; TEST_ADLEN]);
        if expected.trim() != actual.trim() {
            println!("{}", actual);
            assert!(false);
        }
    }

    #[test]
    fn test_argon2i() {
        compare_kats("tests/support/argon2-kats/argon2i",
                     super::Variant::Argon2i);
    }

    #[test]
    fn test_argon2d() {
        compare_kats("tests/support/argon2-kats/argon2d",
                     super::Variant::Argon2d);
    }
}

#[cfg(test)]
mod builders {
    use super::*;
        
    const PASSWORD: &'static str = "hunter4";
    const SALT: &'static str = "salty susan"; 
    
    mod alg {
        use argon2::*;
        
        #[test]
        fn alg_builder() {

            const PASSES: u32 = 5;
            const LANES: u32 = 1;
            const MEMORY_SIZE: u32 = LANES * 8 * 2 + 1;

            let ab = Algorithm::argon2d()
                        .passes(PASSES)
                        .lanes(LANES)
                        .memory_size(MEMORY_SIZE)
                        .hash_length(16);
            
            assert_eq!(PASSES, ab.passes);
            assert_eq!(LANES, ab.lanes);
            assert_eq!(MEMORY_SIZE, ab.memory_size);
            assert_eq!(16, ab.hash_length);
            assert_eq!(Variant::Argon2d, ab.variant);
            
            let ab = ab.variant(Variant::Argon2i);
            assert_eq!(Variant::Argon2i, ab.variant); 
        }
        
         #[test]
        fn bad_lanes() {

            const PASSES: u32 = 5;
            const LANES: u32 = 0;
            const MEMORY_SIZE: u32 = LANES * 8 + 1;
        
            let pb = Algorithm::argon2i()
                        .passes(PASSES)
                        .lanes(LANES)
                        .memory_size(MEMORY_SIZE);
                    
            match pb.build() {
                Result::Err(e) => assert_eq!(e, ParamErr::ZeroLanesSpecified),
                Result::Ok(_) => panic!("Successfully built with no lanes, {}", pb),
            };
        }
        
        #[test]
        fn bad_passes() {
            const PASSES: u32 = 0;
            const LANES: u32 = 1;
            const MEMORY_SIZE: u32 = LANES * 8 * 2 + 1;

            let pb = Algorithm::argon2d()
                        .passes(PASSES)
                        .lanes(LANES)
                        .memory_size(MEMORY_SIZE);

            match pb.build() {
                Result::Err(e) => assert_eq!(e, ParamErr::ZeroPassesSpecified),
                Result::Ok(_) => panic!("Successfully built with passes lanes, {}", pb),
            };
        }
        
        #[test]
        fn bad_memory_size() {
            
            const PASSES: u32 = 5;
            const LANES: u32 = 1;
            const MEMORY_SIZE: u32 = LANES * 8 - 1;
            
            let pb = Algorithm::argon2i()
                        .passes(PASSES)
                        .lanes(LANES)
                        .memory_size(MEMORY_SIZE);
                        
            let exp = ParamErr::TooLittleMemoryForLanes {
                lanes: LANES, 
                memory: MEMORY_SIZE,
            };

            match pb.build() {
                Result::Err(e) => assert_eq!(exp, e),
                Result::Ok(_) => panic!("Successfully built with too little memory, {}", pb),
            };
        }
    }
   
    // Hashing builders    
    mod hash {
        use argon2::*;
        
        fn arg2() -> Argon2 {
            let ab = Algorithm::argon2d().hash_length(16);
            
            Argon2::new(ab.passes, ab.lanes, ab.memory_size, ab.variant)
        }
        
        #[test]
        fn with_secret() {
            
            let data = vec![0xdeu8, 0xad, 0xbe, 0xef];
            
            let mut expected = [0u8; 16];
            arg2().hash(&mut expected, super::PASSWORD.as_bytes(), super::SALT.as_bytes(), &data, &[]);
            
            let ab = Algorithm::argon2d().hash_length(16);
            let mut hb = ab.build().unwrap();
            let hash = hb.secret(&data)
                    .password(super::PASSWORD)
                    .salt(super::SALT)
                    .hash().unwrap();
                    
            assert_eq!(hash, expected);
        }

        #[test]
        fn with_assoc_data() {
            let data = vec![0xdeu8, 0xad, 0xbe, 0xef];
            
            let mut expected = [0u8; 16];
            arg2().hash(&mut expected, super::PASSWORD.as_bytes(), super::SALT.as_bytes(), &[], &data);
            
            
            let ab = Algorithm::argon2d().hash_length(16);
            let mut hb = ab.build().unwrap();
            let hash = hb.assoc_data(&data)
                        .password(super::PASSWORD)
                        .salt(super::SALT)
                        .hash().unwrap();
                    
            assert_eq!(hash, expected);
        }
    }
    
    #[test]
    fn bad_hash_length() {
        const PASSES: u32 = 20;
        const LANES: u32 = 1;
        const MEMORY_SIZE: u32 = LANES * 8 * 2 + 1;

        let pb = Algorithm::argon2d()
                    .passes(PASSES)
                    .lanes(LANES)
                    .memory_size(MEMORY_SIZE)
                    .hash_length(3);

        match pb.build() {
            Result::Err(e) => assert_eq!(e, ParamErr::InvalidHashLength(3)),
            Result::Ok(_) => panic!("Successfully built with bad hash length, {}", pb),
        };
        
        let mut ab = pb.hash_length(20).build().unwrap();
        let ab = ab.password(PASSWORD)
                    .salt(SALT);
        
        let mut bad_vec = vec![0u8; 5];
        let exp_err = ParamErr::InvalidHashBufferLength { buffer_len: 5, hash_length: 20 };
        match ab.hash_inplace(&mut bad_vec) {
            Result::Err(e) => assert_eq!(exp_err, e),
            Result::Ok(_) => panic!("Successfully hashed with buffer too small, {}", ab),
        }
    }
}
