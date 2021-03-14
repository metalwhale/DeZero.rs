use num_traits::Float;
use std::{cell::RefCell, rc::Rc};

struct Variable<'a, D: Float> {
    data: D,
    grad: Option<D>,
    creator: Option<(Rc<RefCell<Variable<'a, D>>>, &'a dyn Function<D>)>,
}
impl<'a, D: Float> Variable<'a, D> {
    fn new(data: D) -> Self {
        Self {
            data,
            grad: None,
            creator: None,
        }
    }

    fn set_creator(&mut self, var: Rc<RefCell<Variable<'a, D>>>, func: &'a dyn Function<D>) {
        self.creator = Some((var, func));
    }

    fn backward(&mut self) {
        if let Some((ref mut x, p)) = self.creator {
            let mut x = x.borrow_mut();
            x.grad = Some(p.backward(x.data, self.grad.unwrap()));
            x.backward();
        }
    }
}

trait Function<D: Float> {
    fn call<'a>(&'a mut self, input: Rc<RefCell<Variable<'a, D>>>) -> Variable<'a, D>
    where
        Self: Sized,
    {
        let x = input.borrow().data;
        let y = self.forward(x);
        let mut output = Variable::new(y);
        output.set_creator(input, self);
        return output;
    }

    fn forward(&self, x: D) -> D;
    fn backward(&self, x: D, gy: D) -> D;
}

#[derive(Default)]
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

#[derive(Default)]
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
    let mut a = Square::default();
    let mut b = Exp::default();
    let mut c = Square::default();
    let x = Rc::new(RefCell::new(Variable::new(0.5)));
    let m = Rc::new(RefCell::new(a.call(x.clone())));
    let n = Rc::new(RefCell::new(b.call(m.clone())));
    let mut y = c.call(n.clone());
    y.grad = Some(1.0);
    y.backward();
    println!("{}", x.clone().borrow().grad.unwrap());
}
