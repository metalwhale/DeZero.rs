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

#[derive(Default)]
struct ExpPass {}
impl Pass for ExpPass {
    fn forward<F: Float>(&self, x: F) -> F {
        return x.exp();
    }
}
type Exp = Function<ExpPass>;

fn main() {
    let a = Square::default();
    let b = Exp::default();
    let c = Square::default();
    let x = Variable { data: 0.5 };
    let a = a.call(x);
    let b = b.call(a);
    let y = c.call(b);
    println!("{}", y.data);
}
