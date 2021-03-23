use num_traits::Float;
use std::cell::Cell;

struct Variable<'a, D: Float> {
    data: D,
    grad: Cell<Option<D>>,
    creator: Option<Calculation<'a, D>>,
}
impl<'a, D: Float> Variable<'a, D> {
    fn new(data: D) -> Self {
        Self {
            data,
            grad: Cell::new(None),
            creator: None,
        }
    }

    fn set_creator(&mut self, input: &'a Variable<'a, D>, function: &'a dyn Function<D>) {
        self.creator = Some(Calculation { input, function });
    }

    fn backward(&self) {
        let mut calcs = vec![];
        let mut x = self;
        let mut grad = match self.grad.get() {
            Some(g) => g,
            None => D::from(1).unwrap(),
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

trait Function<D: Float> {
    fn call<'a>(&'a self, input: &'a Variable<'a, D>) -> Variable<'a, D>
    where
        Self: Sized,
    {
        let x = input.data;
        let y = self.forward(x);
        let mut output = Variable::new(y);
        output.set_creator(input, self);
        return output;
    }

    fn forward(&self, x: D) -> D;
    fn backward(&self, x: D, gy: D) -> D;
}

struct Calculation<'a, D: Float> {
    input: &'a Variable<'a, D>,
    function: &'a dyn Function<D>,
}

struct Square {}
impl<D: Float> Function<D> for Square {
    fn forward(&self, x: D) -> D {
        return x.powi(2);
    }

    fn backward(&self, x: D, gy: D) -> D {
        let gx = D::from(2).unwrap() * x * gy;
        return gx;
    }
}

struct Exp {}
impl<D: Float> Function<D> for Exp {
    fn forward(&self, x: D) -> D {
        return x.exp();
    }

    fn backward(&self, x: D, gy: D) -> D {
        let gx = x.exp() * gy;
        return gx;
    }
}

fn square<'a, D: Float>(x: &'a Variable<'a, D>) -> Variable<'a, D> {
    return Square {}.call(x);
}

fn exp<'a, D: Float>(x: &'a Variable<'a, D>) -> Variable<'a, D> {
    return Exp {}.call(x);
}

fn main() {
    let x = Variable::new(0.5);
    square(&exp(&square(&x))).backward();
    println!("{}", x.grad.get().unwrap());
}
