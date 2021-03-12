use num_traits::Float;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

struct Variable<'a, D: Float> {
    data: D,
    grad: Option<D>,
    creator: Option<(Rc<RefCell<Variable<'a, D>>>, &'a dyn Pass<D>)>,
}
impl<'a, D: Float> Variable<'a, D> {
    fn new(data: D) -> Self {
        Self {
            data,
            grad: None,
            creator: None,
        }
    }

    fn set_creator(&mut self, func: (Rc<RefCell<Variable<'a, D>>>, &'a dyn Pass<D>)) {
        self.creator = Some(func);
    }

    fn backward(&mut self) {
        if let Some((ref mut x, p)) = self.creator {
            let mut x = x.borrow_mut();
            x.grad = Some(p.backward(x.data, self.grad.unwrap()));
            x.backward();
        }
    }
}

trait Pass<D: Float> {
    fn forward(&self, x: D) -> D;
    fn backward(&self, x: D, gy: D) -> D;
}

struct Function<D: Float, P: Pass<D>> {
    pass: P,
    phantom: PhantomData<D>,
}
impl<D: Float, P: Pass<D>> Function<D, P> {
    fn call<'a>(&'a mut self, input: Rc<RefCell<Variable<'a, D>>>) -> Variable<'a, D> {
        let x = input.borrow().data;
        let y = self.pass.forward(x);
        let mut output = Variable::new(y);
        output.set_creator((input, &self.pass));
        return output;
    }
}
impl<D: Float + Default, P: Pass<D> + Default> Default for Function<D, P> {
    fn default() -> Self {
        Function {
            pass: Default::default(),
            phantom: PhantomData,
        }
    }
}

#[derive(Default)]
struct Square {}
impl<D: Float> Pass<D> for Square {
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
impl<D: Float> Pass<D> for Exp {
    fn forward(&self, x: D) -> D {
        return x.exp();
    }

    fn backward(&self, x: D, gy: D) -> D {
        let gx = x.exp() * gy;
        return gx;
    }
}

fn main() {
    let mut a = Function::<f64, Square>::default();
    let mut b = Function::<f64, Exp>::default();
    let mut c = Function::<f64, Square>::default();
    let x = Rc::new(RefCell::new(Variable::new(0.5)));
    let m = Rc::new(RefCell::new(a.call(x.clone())));
    let n = Rc::new(RefCell::new(b.call(m)));
    let mut y = c.call(n);
    y.grad = Some(1.0);
    y.backward();
    println!("{}", x.clone().borrow().grad.unwrap());
}
