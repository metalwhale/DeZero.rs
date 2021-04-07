use ndarray::{Array, ArrayD, Dimension};
use num_traits::Float;
use std::{cell::RefCell, rc::Rc};

struct VariableInternal<'c, N: Float> {
    data: ArrayD<N>,
    grad: Option<ArrayD<N>>,
    creator: Option<Rc<Calculation<'c, N>>>,
}
impl<'c, N: Float> VariableInternal<'c, N> {
    fn new<D: Dimension>(data: Array<N, D>) -> Self {
        Self {
            data: data.into_dyn(),
            grad: None,
            creator: None,
        }
    }
}

pub struct Variable<'c, N: Float> {
    internal: Rc<RefCell<VariableInternal<'c, N>>>,
}
impl<'c, N: Float> Variable<'c, N> {
    fn new<D: Dimension>(data: Array<N, D>) -> Self {
        Self {
            internal: Rc::new(RefCell::new(VariableInternal::new(data))),
        }
    }

    fn data(&self) -> ArrayD<N> {
        self.internal.borrow().data.clone()
    }

    fn grad(&self) -> Option<ArrayD<N>> {
        self.internal.borrow().grad.clone()
    }

    pub fn backward(&self) {
        let mut inputs = vec![Rc::clone(&self.internal)];
        let mut gxs = vec![match self.grad() {
            Some(g) => g,
            None => ArrayD::ones(self.internal.borrow().data.raw_dim()),
        }];
        let mut calcs = vec![];
        loop {
            inputs.iter().zip(gxs.into_iter()).for_each(|(i, gx)| {
                i.borrow_mut().grad = Some(gx.clone());
                if let Some(creator) = &i.borrow().creator {
                    calcs.push((Rc::clone(creator), gx));
                }
            });
            if let Some((c, gy)) = calcs.pop() {
                inputs = c.inputs.clone();
                gxs = c.function.backward(
                    &inputs
                        .iter()
                        .map(|i| i.borrow().data.clone())
                        .collect::<Vec<_>>(),
                    &gy,
                );
            } else {
                break;
            }
        }
    }
}

trait Function<N: Float> {
    fn call<'c>(self, inputs: &[&Variable<'c, N>]) -> Variable<'c, N>
    where
        Self: 'c + Sized,
    {
        let xs = inputs.iter().map(|i| i.data()).collect::<Vec<_>>();
        let y = self.forward(&xs);
        let output = Variable::new(y);
        output.internal.borrow_mut().creator = Some(Rc::new(Calculation {
            inputs: inputs.iter().map(|i| Rc::clone(&i.internal)).collect(),
            function: Box::new(self),
        }));
        output
    }

    fn forward(&self, xs: &[ArrayD<N>]) -> ArrayD<N>;
    fn backward(&self, xs: &[ArrayD<N>], gy: &ArrayD<N>) -> Vec<ArrayD<N>>;
}

struct Calculation<'c, N: Float> {
    inputs: Vec<Rc<RefCell<VariableInternal<'c, N>>>>,
    function: Box<dyn 'c + Function<N>>,
}

struct Square;
impl<N: Float> Function<N> for Square {
    fn forward(&self, xs: &[ArrayD<N>]) -> ArrayD<N> {
        xs[0].mapv(|n| n.powi(2))
    }

    fn backward(&self, xs: &[ArrayD<N>], gy: &ArrayD<N>) -> Vec<ArrayD<N>> {
        let x = &xs[0];
        let two = N::from(2).unwrap();
        let gx = x.mapv(|n| two * n) * gy;
        vec![gx]
    }
}
pub fn square<'c, N: Float>(x: &Variable<'c, N>) -> Variable<'c, N> {
    Square.call(&[x])
}

struct Add;
impl<N: Float> Function<N> for Add {
    fn forward(&self, xs: &[ArrayD<N>]) -> ArrayD<N> {
        &xs[0] + &xs[1]
    }

    fn backward(&self, _xs: &[ArrayD<N>], gy: &ArrayD<N>) -> Vec<ArrayD<N>> {
        vec![gy.clone(), gy.clone()]
    }
}
pub fn add<'c, N: Float>(x0: &Variable<'c, N>, x1: &Variable<'c, N>) -> Variable<'c, N> {
    Add.call(&[x0, x1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;

    #[test]
    fn test() {
        let x = Variable::new(arr0(2.0));
        let y = Variable::new(arr0(3.0));
        let z = add(&square(&x), &square(&y));
        z.backward();
        println!("{}", z.data());
        println!("{}", x.grad().unwrap());
        println!("{}", y.grad().unwrap());
    }
}
