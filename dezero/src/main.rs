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

fn numerical_diff<D: Float, F: Fn(Variable<D>) -> Variable<D>>(f: F, x: Variable<D>, eps: D) -> D {
    let x0 = Variable { data: x.data - eps };
    let x1 = Variable { data: x.data + eps };
    let y0 = f(x0);
    let y1 = f(x1);
    return (y1.data - y0.data) / (D::from(2).unwrap() * eps);
}

fn main() {
    const EPS: f64 = 1e-4;
    let a = Function::<Square>::default();
    let x = Variable { data: 2.0 };
    let dy = numerical_diff(|v| a.call(v), x, EPS);
    println!("{}", dy);

    fn f<D: Float>(x: Variable<D>) -> Variable<D> {
        let a = Function::<Square>::default();
        let b = Function::<Exp>::default();
        let c = Function::<Square>::default();
        return c.call(b.call(a.call(x)));
    }
    let x = Variable { data: 0.5 };
    let dy = numerical_diff(f, x, EPS);
    println!("{}", dy);
}
