// a comment
pub fn alpha(x: i32) -> i32 {
    x + 1
}

struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn beta(&self) -> i32 {
        self.x
    }
}
