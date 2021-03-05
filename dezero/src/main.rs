use num_traits::Float;

struct Variable<F: Float> {
    data: F,
}

fn main() {
    let data = 1.0;
    let mut x = Variable { data };
    println!("{}", x.data);
    x.data = 2.0;
    println!("{}", x.data);
}
