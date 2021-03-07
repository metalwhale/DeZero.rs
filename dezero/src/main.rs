use num_traits::Float;

#[derive(Default, Copy, Clone)]
struct Variable<D: Float> {
    data: D,
    grad: Option<D>,
}
impl<D: Float> Variable<D> {
    fn new(data: D) -> Self {
        Self { data, grad: None }
    }
}

trait Pass {
    fn forward<D: Float>(&self, x: D) -> D;
    fn backward<D: Float>(&self, x: D, gy: D) -> D;
}

struct Function<D: Float, P: Pass> {
    input: Variable<D>,
    pass: P,
}
impl<D: Float, P: Pass> Function<D, P> {
    fn call(&mut self, input: Variable<D>) -> Variable<D> {
        let x = input.data;
        let y = self.pass.forward(x);
        let output = Variable::new(y);
        self.input = input;
        return output;
    }
}
impl<D: Float + Default, P: Pass + Default> Default for Function<D, P> {
    fn default() -> Self {
        Function {
            input: Default::default(),
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

    fn backward<D: Float>(&self, x: D, gy: D) -> D {
        let gx = D::from(2).unwrap() * x * gy;
        return gx;
    }
}

#[derive(Default)]
struct Exp {}
impl Pass for Exp {
    fn forward<D: Float>(&self, x: D) -> D {
        return x.exp();
    }

    fn backward<D: Float>(&self, x: D, gy: D) -> D {
        let gx = x.exp() * gy;
        return gx;
    }
}

fn main() {
    let mut a = Function::<f64, Square>::default();
    let mut b = Function::<f64, Exp>::default();
    let mut c = Function::<f64, Square>::default();
    let mut x = Variable::new(0.5);
    let mut m = a.call(x);
    let mut n = b.call(m);
    let mut y = c.call(n);
    y.grad = Some(1.0);
    n.grad = Some(c.pass.backward(c.input.data, y.grad.unwrap()));
    m.grad = Some(b.pass.backward(b.input.data, n.grad.unwrap()));
    x.grad = Some(a.pass.backward(a.input.data, m.grad.unwrap()));
    println!("{}", x.grad.unwrap());
}
