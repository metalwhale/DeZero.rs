use ndarray::{arr0, ArrayD};
use num_traits::Float;
use std::cell::RefCell;

struct Variable<'c, N: Float> {
    data: ArrayD<N>,
    grad: RefCell<Option<ArrayD<N>>>,
    creator: Option<Calculation<'c, N>>,
}
impl<'c, N: Float> Variable<'c, N> {
    fn new(data: ArrayD<N>, creator: Option<Calculation<'c, N>>) -> Self {
        Self {
            data,
            grad: RefCell::new(None),
            creator,
        }
    }

    fn backward(&self) {
        let mut calcs = vec![];
        let mut x = self;
        let mut grad = match self.grad.borrow().clone() {
            Some(g) => g,
            None => ArrayD::ones(x.data.raw_dim()),
        };
        loop {
            x.grad.replace(Some(grad.clone()));
            if let Some(creator) = &x.creator {
                calcs.push((creator, grad));
            }
            if let Some((Calculation { input, function: f }, gy)) = calcs.pop() {
                x = input;
                grad = f.backward(&x.data, &gy);
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
        let x = &input.data;
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

    fn forward(&self, x: &ArrayD<N>) -> ArrayD<N>;
    fn backward(&self, x: &ArrayD<N>, gy: &ArrayD<N>) -> ArrayD<N>;
}

struct Calculation<'c, N: Float> {
    input: &'c Variable<'c, N>,
    function: &'c dyn Function<N>,
}

struct Square;
impl<N: Float> Function<N> for Square {
    fn forward(&self, x: &ArrayD<N>) -> ArrayD<N> {
        return x.mapv(|n| n.powi(2));
    }

    fn backward(&self, x: &ArrayD<N>, gy: &ArrayD<N>) -> ArrayD<N> {
        let two = N::from(2).unwrap();
        let gx = x.mapv(|n| two * n) * gy;
        return gx;
    }
}

struct Exp;
impl<N: Float> Function<N> for Exp {
    fn forward(&self, x: &ArrayD<N>) -> ArrayD<N> {
        return x.mapv(N::exp);
    }

    fn backward(&self, x: &ArrayD<N>, gy: &ArrayD<N>) -> ArrayD<N> {
        let gx = x.mapv(N::exp) * gy;
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
    let x = Variable::new(arr0(0.5).into_dyn(), None);
    square(&exp(&square(&x))).backward();
    println!("{}", x.grad.borrow().as_ref().unwrap());
}
