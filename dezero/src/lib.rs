use ndarray::{Array, ArrayD, Dimension};
use num_traits::Float;
use std::cell::RefCell;

pub struct Variable<'c, N: Float> {
    data: ArrayD<N>,
    grad: RefCell<Option<ArrayD<N>>>,
    creator: Option<Calculation<'c, N>>,
}
impl<'c, N: Float> Variable<'c, N> {
    pub fn new<D: Dimension>(data: Array<N, D>, creator: Option<Calculation<'c, N>>) -> Self {
        Self {
            data: data.into_dyn(),
            grad: RefCell::new(None),
            creator,
        }
    }

    pub fn backward(&self) {
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

pub struct Calculation<'c, N: Float> {
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

pub fn square<'c, N: Float>(x: &'c Variable<'c, N>) -> Variable<'c, N> {
    return Square.call(x);
}

pub fn exp<'c, N: Float>(x: &'c Variable<'c, N>) -> Variable<'c, N> {
    return Exp.call(x);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn numerical_diff<N: Float, F: for<'c> Fn(&'c Variable<'c, N>) -> Variable<'c, N>>(
        f: F,
        x: &Variable<N>,
        eps: N,
    ) -> ArrayD<N> {
        let x0 = Variable::new(x.data.mapv(|n| n - eps), None);
        let x1 = Variable::new(x.data.mapv(|n| n + eps), None);
        let y0 = f(&x0);
        let y1 = f(&x1);
        let two = N::from(2).unwrap();
        return (y1.data - y0.data).mapv(|n| n / (two * eps));
    }

    #[test]
    fn square_forward() {
        let x = Variable::new(arr0(2.0), None);
        let y = square(&x);
        let expected = arr0(4.0).into_dyn();
        assert_eq!(&y.data, &expected);
    }

    #[test]
    fn square_backward() {
        let x = Variable::new(arr0(3.0), None);
        let y = square(&x);
        y.backward();
        let expected = arr0(6.0).into_dyn();
        assert_eq!(x.grad.borrow().as_ref().unwrap(), &expected);
    }

    #[test]
    fn square_gradient_check() {
        let x = Variable::new(Array::random((), Uniform::new(0.0, 1.0)), None);
        let y = square(&x);
        y.backward();
        let num_grad = numerical_diff(square, &x, 1e-4);
        let flg = x
            .grad
            .borrow()
            .as_ref()
            .unwrap()
            .abs_diff_eq(&num_grad, 1e-8);
        assert!(flg);
    }
}
