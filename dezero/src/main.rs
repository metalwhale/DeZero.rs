use num_traits::Float;

struct Variable<F: Float> {
    data: F,
}

trait Pass {
    fn forward<F: Float>(&self, x: F) -> F;
}

struct Function<P: Pass> {
    pass: P,
}
impl<P> Function<P>
where
    P: Pass,
{
    fn call<F: Float>(&self, input: Variable<F>) -> Variable<F> {
        let x = input.data;
        let y = self.pass.forward(x);
        let output = Variable { data: y };
        return output;
    }
}
impl<P> Default for Function<P>
where
    P: Pass + Default,
{
    fn default() -> Self {
        Function {
            pass: Default::default(),
        }
    }
}

#[derive(Default)]
struct SquarePass {}
impl Pass for SquarePass {
    fn forward<F: Float>(&self, x: F) -> F {
        return x.powi(2);
    }
}
type Square = Function<SquarePass>;

fn main() {
    let x = Variable { data: 10.0 };
    let f = Square::default();
    let y = f.call(x);
    println!("{}", y.data);
}
