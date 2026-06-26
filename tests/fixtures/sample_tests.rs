pub fn production_fn(x: i32) -> i32 {
    x * 2
}

pub struct Widget {
    pub id: u32,
}

#[cfg(test)]
fn test_helper() -> i32 {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_widget() {
        assert_eq!(Widget { id: 1 }.id, 1);
    }

    #[test]
    fn checks_production_fn() {
        assert_eq!(production_fn(2), 4);
    }
}

#[test]
fn standalone_test() {
    assert!(production_fn(1) == 2);
}
