use num_traits::Float;

struct Variable<D: Float> {
    data: D,
}

trait Pass {
    fn forward<D: Float>(&self, x: D) -> D;
}

struct Function<P: Pass> {
    pass: P,
}
impl<P: Pass> Function<P> {
    fn call<D: Float>(&self, input: Variable<D>) -> Variable<D> {
        let x = input.data;
        let y = self.pass.forward(x);
        let output = Variable { data: y };
        return output;
    }
}
impl<P: Pass + Default> Default for Function<P> {
    fn default() -> Self {
        Function {
            pass: Default::default(),
        }
    }
}

#[derive(Default)]
struct Square {}
impl Pass for Square {
    fn forward<D: Float>(&self, x: D) -> D {
        return x.powi(2);
    }
}

#[derive(Default)]
struct Exp {}
impl Pass for Exp {
    fn forward<D: Float>(&self, x: D) -> D {
        return x.exp();
    }
}

fn main() {
    let a = Function::<Square>::default();
    let b = Function::<Exp>::default();
    let c = Function::<Square>::default();
    let x = Variable { data: 0.5 };
    let a = a.call(x);
    let b = b.call(a);
    let y = c.call(b);
    println!("{}", y.data);
}
