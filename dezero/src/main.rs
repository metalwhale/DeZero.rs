use num_traits::Float;
use std::cell::Cell;

struct Variable<'c, N: Float> {
    data: N,
    grad: Cell<Option<N>>,
    creator: Option<Calculation<'c, N>>,
}
impl<'c, N: Float> Variable<'c, N> {
    fn new(data: N, creator: Option<Calculation<'c, N>>) -> Self {
        Self {
            data,
            grad: Cell::new(None),
            creator,
        }
    }

    fn backward(&self) {
        let mut calcs = vec![];
        let mut x = self;
        let mut grad = match self.grad.get() {
            Some(g) => g,
            None => N::one(),
        };
        loop {
            x.grad.set(Some(grad));
            if let Some(creator) = &x.creator {
                calcs.push((creator, grad));
            }
            if let Some((Calculation { input, function: f }, gy)) = calcs.pop() {
                x = input;
                grad = f.backward(x.data, gy);
            } else {
                break;
            }
        }
    }
}

trait Function<N: Float> {
    fn call<'c>(&'c self, input: &'c Variable<'c, N>) -> Variable<'c, N>
    where
        Self: Sized,
    {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::new(
            y,
            Some(Calculation {
                input,
                function: self,
            }),
        );
        return output;
    }

    fn forward(&self, x: N) -> N;
    fn backward(&self, x: N, gy: N) -> N;
}

struct Calculation<'c, N: Float> {
    input: &'c Variable<'c, N>,
    function: &'c dyn Function<N>,
}

struct Square;
impl<N: Float> Function<N> for Square {
    fn forward(&self, x: N) -> N {
        return x.powi(2);
    }

    fn backward(&self, x: N, gy: N) -> N {
        let gx = N::from(2).unwrap() * x * gy;
        return gx;
    }
}

struct Exp;
impl<N: Float> Function<N> for Exp {
    fn forward(&self, x: N) -> N {
        return x.exp();
    }

    fn backward(&self, x: N, gy: N) -> N {
        let gx = x.exp() * gy;
        return gx;
    }
}

fn square<'c, N: Float>(x: &'c Variable<'c, N>) -> Variable<'c, N> {
    return Square.call(x);
}

fn exp<'c, N: Float>(x: &'c Variable<'c, N>) -> Variable<'c, N> {
    return Exp.call(x);
}

fn main() {
    let x = Variable::new(0.5, None);
    square(&exp(&square(&x))).backward();
    println!("{}", x.grad.get().unwrap());
}
