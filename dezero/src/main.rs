use num_traits::Float;
use std::{cell::RefCell, rc::Rc};

struct Variable<'a, D: Float> {
    data: D,
    grad: Option<D>,
    creator: Option<Calculation<'a, D>>,
}
impl<'a, D: Float> Variable<'a, D> {
    fn new(data: D) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            data,
            grad: None,
            creator: None,
        }))
    }

    fn set_creator(&mut self, input: Rc<RefCell<Variable<'a, D>>>, function: &'a dyn Function<D>) {
        self.creator = Some(Calculation { input, function });
    }

    fn backward(&self) -> Result<(), ()> {
        match self.creator {
            Some(Calculation {
                input: ref x,
                function: f,
            }) => match self.grad {
                Some(grad) => {
                    let mut x = x.borrow_mut();
                    x.grad = Some(f.backward(x.data, grad));
                    x.backward()
                }
                None => Err(()),
            },
            None => Ok(()),
        }
    }
}

trait Function<D: Float> {
    fn call<'a>(&'a self, input: &Rc<RefCell<Variable<'a, D>>>) -> Rc<RefCell<Variable<'a, D>>>
    where
        Self: Sized,
    {
        let input = Rc::clone(input);
        let x = input.borrow().data;
        let y = self.forward(x);
        let output = Variable::new(y);
        output.borrow_mut().set_creator(input, self);
        return output;
    }

    fn forward(&self, x: D) -> D;
    fn backward(&self, x: D, gy: D) -> D;
}

struct Calculation<'a, D: Float> {
    input: Rc<RefCell<Variable<'a, D>>>,
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

fn main() {
    let a = Square {};
    let b = Exp {};
    let c = Square {};
    let x = Variable::new(0.5);
    let m = a.call(&x);
    let n = b.call(&m);
    let y = c.call(&n);
    let mut y = y.borrow_mut();
    y.grad = Some(1.0);
    if let Ok(_) = y.backward() {
        println!("{}", x.borrow().grad.unwrap());
    }
}
