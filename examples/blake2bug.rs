extern crate crypto;
extern crate blake2_rfc;

use crypto::blake2b as b2;
use crypto::digest::Digest;
use blake2_rfc::blake2b as rb2;

fn main() {
    let small = [3, 1, 2, 4];
    let mut fail = [0; 64];

    for i in 250..255 {
        let large = (0..i).collect::<Vec<u8>>();

        let mut rustcrypto = b2::Blake2b::new(64);
        let mut rblake2 = rb2::Blake2b::new(64);

        rustcrypto.input(&small);
        rustcrypto.input(&large[..]);
        rustcrypto.result(&mut fail);

        rblake2.update(&small);
        rblake2.update(&large[..]);
        let correct = rblake2.finalize();

        println!("hashes for [3, 1, 2, 4] + w where w.len() = {}", large.len());
        printu8(&fail);
        printu8(correct.as_bytes());
        assert_eq!(&fail[..], correct.as_bytes());
    }
}

fn printu8(bs: &[u8]) {
    for b in bs.iter() {
        print!("{:02x}", b);
    }
    println!("");
}
