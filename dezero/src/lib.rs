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
        let mut inputs = vec![self];
        let mut gxs = vec![match self.grad.borrow().clone() {
            Some(g) => g,
            None => ArrayD::ones(self.data.raw_dim()),
        }];
        let mut calcs = vec![];
        loop {
            inputs.iter().zip(gxs.into_iter()).for_each(|(i, gx)| {
                i.grad.replace(Some(gx.clone()));
                if let Some(creator) = &i.creator {
                    calcs.push((creator, gx));
                }
            });
            if let Some((c, gy)) = calcs.pop() {
                inputs = c.inputs.to_vec();
                gxs = c
                    .function
                    .backward(&inputs.iter().map(|i| &i.data).collect::<Vec<_>>(), &gy);
            } else {
                break;
            }
        }
    }
}

trait Function<N: Float> {
    fn call<'c>(&'c self, inputs: &[&'c Variable<'c, N>]) -> Variable<'c, N>
    where
        Self: Sized,
    {
        let xs = inputs.iter().map(|i| &i.data).collect::<Vec<_>>();
        let y = self.forward(&xs);
        let output = Variable::new(
            y,
            Some(Calculation {
                inputs: inputs.to_vec(),
                function: self,
            }),
        );
        return output;
    }

    fn forward(&self, xs: &[&ArrayD<N>]) -> ArrayD<N>;
    fn backward(&self, xs: &[&ArrayD<N>], gy: &ArrayD<N>) -> Vec<ArrayD<N>>;
}

pub struct Calculation<'c, N: Float> {
    inputs: Vec<&'c Variable<'c, N>>,
    function: &'c dyn Function<N>,
}

struct Square;
impl<N: Float> Function<N> for Square {
    fn forward(&self, xs: &[&ArrayD<N>]) -> ArrayD<N> {
        let y = xs[0].mapv(|n| n.powi(2));
        return y;
    }

    fn backward(&self, xs: &[&ArrayD<N>], gy: &ArrayD<N>) -> Vec<ArrayD<N>> {
        let x = xs[0];
        let two = N::from(2).unwrap();
        let gx = x.mapv(|n| two * n) * gy;
        return vec![gx];
    }
}
pub fn square<'c, N: Float>(x: &'c Variable<'c, N>) -> Variable<'c, N> {
    return Square.call(&[x]);
}

struct Add;
impl<N: Float> Function<N> for Add {
    fn forward(&self, xs: &[&ArrayD<N>]) -> ArrayD<N> {
        let y = xs[0] + xs[1];
        return y;
    }

    fn backward(&self, _xs: &[&ArrayD<N>], gy: &ArrayD<N>) -> Vec<ArrayD<N>> {
        return vec![gy.clone(), gy.clone()];
    }
}
pub fn add<'c, N: Float>(x0: &'c Variable<'c, N>, x1: &'c Variable<'c, N>) -> Variable<'c, N> {
    return Add.call(&[x0, x1]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;

    #[test]
    fn test() {
        let x = Variable::new(arr0(2.0), None);
        let y = Variable::new(arr0(3.0), None);
        let sx = square(&x);
        let sy = square(&y);
        let z = add(&sx, &sy);
        z.backward();
        println!("{}", z.data);
        println!("{}", x.grad.borrow().as_ref().unwrap());
        println!("{}", y.grad.borrow().as_ref().unwrap());
    }
}