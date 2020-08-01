use ggez::{graphics::Color, nalgebra as na, Context, GameResult};

fn main() -> GameResult<()> {
    use ggez::conf::*;
    better_panic::install();
    pretty_env_logger::init();
    let (mut ctx, mut evl) = ggez::ContextBuilder::new("blocks_hanging_out", "michcioperz")
        .window_setup(WindowSetup {
            title: "blocks hanging out".to_owned(),
            ..WindowSetup::default()
        })
        .window_mode(WindowMode {
            resizable: true,
            ..WindowMode::default()
        })
        .build()
        .unwrap();
    let mut game = Game::new(&mut ctx)?;
    ggez::event::run(&mut ctx, &mut evl, &mut game)
}

trait Darken {
    fn darken(&self) -> Self;
    fn ghost(&self) -> Self;
}

impl Darken for Color {
    fn darken(&self) -> Color {
        let mut hsl: palette::Hsl = palette::Srgb::new(self.r, self.g, self.b).into();
        hsl.lightness = (hsl.lightness * 0.7f32).max(0f32);
        let rgb: palette::Srgb = hsl.into();
        let (r, g, b): (u8, u8, u8) = rgb.into_format().into_components();
        Color::from_rgba(r, g, b, 50)
    }

    fn ghost(&self) -> Color {
        let mut hsl: palette::Hsl = palette::Srgb::new(self.r, self.g, self.b).into();
        hsl.lightness = (hsl.lightness * 1.2f32).min(1f32);
        let rgb: palette::Srgb = hsl.into();
        let (r, g, b): (u8, u8, u8) = rgb.into_format().into_components();
        Color::from_rgba(r, g, b, 50)
    }
}

#[derive(Debug)]
struct Palette {
    board: Color,
    bg: Color,
    pieces: Vec<Color>,
    grid: Color,
}

impl Default for Palette {
    fn default() -> Palette {
        Palette {
            board: Color::from_rgb(180, 210, 186),
            bg: Color::from_rgb(52, 54, 51),
            pieces: vec![
                Color::from_rgb(245, 208, 197),
                Color::from_rgb(214, 159, 126),
                Color::from_rgb(119, 73, 54),
                Color::from_rgb(60, 0, 0),
            ],
            grid: Color::from_rgb(0, 0, 0), // TODO
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Border {
    top: bool,
    right: bool,
    bottom: bool,
    left: bool,
}

macro_rules! border {
    {$( $x:ident ),*} => { Border { $( $x: true, )* ..Border::default() } };
}

#[derive(Clone, Copy, Debug)]
struct Field {
    color: Color,
    border: Border,
}

impl Field {
    #[inline]
    fn mesh(
        self,
        mb: &mut ggez::graphics::MeshBuilder,
        pos: Position,
        tile_size: f32,
        screen_offset: Offset,
    ) -> &mut ggez::graphics::MeshBuilder {
        use ggez::graphics;
        let mut m = mb;
        let x = screen_offset.x as f32 + tile_size * pos.x as f32;
        let y = screen_offset.y as f32 + tile_size * pos.y as f32;
        m = m.rectangle(
            graphics::DrawMode::fill(),
            graphics::Rect {
                x,
                y,
                w: tile_size,
                h: tile_size,
            },
            self.color,
        );
        let border_color = self.color.darken();
        if self.border.top {
            m = m
                .line(
                    &[na::Point2::new(x, y), na::Point2::new(x + tile_size, y)],
                    tile_size / 5f32,
                    border_color,
                )
                .unwrap();
        }
        if self.border.left {
            m = m
                .line(
                    &[na::Point2::new(x, y), na::Point2::new(x, y + tile_size)],
                    tile_size / 5f32,
                    border_color,
                )
                .unwrap();
        }
        if self.border.bottom {
            m = m
                .line(
                    &[
                        na::Point2::new(x, y + tile_size),
                        na::Point2::new(x + tile_size, y + tile_size),
                    ],
                    tile_size / 5f32,
                    border_color,
                )
                .unwrap();
        }
        if self.border.right {
            m = m
                .line(
                    &[
                        na::Point2::new(x + tile_size, y),
                        na::Point2::new(x + tile_size, y + tile_size),
                    ],
                    tile_size / 5f32,
                    border_color,
                )
                .unwrap();
        }
        m
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Position {
    x: isize,
    y: isize,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Offset {
    x: isize,
    y: isize,
}

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
enum Rotation {
    Verbatim,
    NinetyClockwise,
    UpsideDown,
    NinetyCounterclockwise,
}

impl Position {
    fn distance(&self, other: &Position) -> usize {
        (self.x.max(other.x) - self.x.min(other.x) + self.y.max(other.y) - self.y.min(other.y))
            as usize
    }
}

impl Rotation {
    const COUNT: usize = 4;

    fn distance(&self, other: &Rotation) -> usize {
        let l: usize = self.into();
        let r: usize = other.into();
        l.max(r) - l.max(l)
    }
}

impl From<usize> for Rotation {
    fn from(i: usize) -> Rotation {
        use Rotation::*;
        match i.wrapping_rem(4) {
            0 => Verbatim,
            1 => NinetyClockwise,
            2 => UpsideDown,
            3 => NinetyCounterclockwise,
            _ => unreachable!(),
        }
    }
}

impl From<&Rotation> for usize {
    fn from(r: &Rotation) -> usize {
        use Rotation::*;
        match r {
            Verbatim => 0,
            NinetyClockwise => 1,
            UpsideDown => 2,
            NinetyCounterclockwise => 3,
        }
    }
}

impl std::ops::Add<Rotation> for Rotation {
    type Output = Rotation;
    fn add(self, other: Rotation) -> Rotation {
        Rotation::from(usize::from(&self).wrapping_add(usize::from(&other)))
    }
}

impl std::ops::Sub<Rotation> for Rotation {
    type Output = Rotation;
    fn sub(self, other: Rotation) -> Rotation {
        Rotation::from(usize::from(&self).wrapping_sub(usize::from(&other)))
    }
}

impl std::ops::Neg for Rotation {
    type Output = Rotation;
    fn neg(self) -> Rotation {
        use Rotation::*;
        match self {
            NinetyClockwise => NinetyCounterclockwise,
            NinetyCounterclockwise => NinetyClockwise,
            _ => self,
        }
    }
}

impl std::ops::Neg for Offset {
    type Output = Offset;
    fn neg(self) -> Offset {
        Offset {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl std::ops::Mul<&Rotation> for &Border {
    type Output = Border;
    fn mul(self, other: &Rotation) -> Border {
        use Rotation::*;
        match other {
            Verbatim => self.clone(),
            UpsideDown => Border {
                top: self.bottom,
                bottom: self.top,
                left: self.right,
                right: self.left,
            },
            NinetyClockwise => Border {
                top: self.left,
                left: self.bottom,
                bottom: self.right,
                right: self.top,
            },
            NinetyCounterclockwise => Border {
                top: self.right,
                right: self.bottom,
                bottom: self.left,
                left: self.top,
            },
        }
    }
}

impl std::ops::Mul<&Rotation> for &Offset {
    type Output = Offset;
    fn mul(self, other: &Rotation) -> Offset {
        use Rotation::*;
        match other {
            Verbatim => *self,
            NinetyClockwise => Offset {
                x: -self.y,
                y: self.x,
            },
            UpsideDown => Offset {
                x: -self.x,
                y: -self.y,
            },
            NinetyCounterclockwise => Offset {
                x: self.y,
                y: -self.x,
            },
        }
    }
}

impl std::ops::Add<Offset> for Position {
    type Output = Position;
    fn add(self, other: Offset) -> Position {
        Position {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Position {
    fn saturating_add(self, other: Offset) -> Position {
        Position {
            x: (self.x + other.x).min(Board::WIDTH as isize - 1).max(0),
            y: (self.y + other.y).min(Board::HEIGHT as isize - 1).max(0),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum PieceShape {
    I,
    J,
    L,
    O,
    S,
    T,
    Z,
}

impl PieceShape {
    const COUNT: u8 = 7;

    #[inline]
    fn offsets(&self) -> [Offset; 4] {
        use PieceShape::*;
        macro_rules! off0 {
            [ $( $x:expr ),* ] => { [Offset {x: 0, y: 0}, $( $x, )* ] };
        }
        match self {
            I => off0![
                Offset { x: 1, y: 0 },
                Offset { x: -1, y: 0 },
                Offset { x: -2, y: 0 }
            ],
            J => off0![
                Offset { x: 0, y: -1 },
                Offset { x: 0, y: 1 },
                Offset { x: -1, y: 1 }
            ],
            L => off0![
                Offset { x: 0, y: -1 },
                Offset { x: 0, y: 1 },
                Offset { x: 1, y: 1 }
            ],
            O => off0![
                Offset { x: 1, y: 0 },
                Offset { x: 0, y: 1 },
                Offset { x: 1, y: 1 }
            ],
            S => off0![
                Offset { x: 1, y: 0 },
                Offset { x: 0, y: 1 },
                Offset { x: -1, y: 1 }
            ],
            T => off0![
                Offset { x: 1, y: 0 },
                Offset { x: -1, y: 0 },
                Offset { x: 0, y: -1 }
            ],
            Z => off0![
                Offset { x: -1, y: 0 },
                Offset { x: 0, y: 1 },
                Offset { x: 1, y: 1 }
            ],
        }
    }

    #[inline]
    fn borders(&self) -> [Border; 4] {
        use PieceShape::*;
        match self {
            I => [
                border! {top, bottom},
                border! {top, bottom, right},
                border! {top, bottom},
                border! {top, bottom, left},
            ],
            J => [
                border! {left, right},
                border! {left, right, top},
                border! {bottom, right},
                border! {top, left, bottom},
            ],
            L => [
                border! {left, right},
                border! {left, right, top},
                border! {bottom, left},
                border! {top, right, bottom},
            ],
            O => [
                border! {left, top},
                border! {top, right},
                border! {left, bottom},
                border! {bottom, right},
            ],
            S => [
                border! {left, top},
                border! {top, right, bottom},
                border! {right, bottom},
                border! {top, left, bottom},
            ],
            T => [
                border! {bottom},
                border! {top, right, bottom},
                border! {top, left, bottom},
                border! {left, top, right},
            ],
            Z => [
                border! {right, top},
                border! {top, left, bottom},
                border! {left, bottom},
                border! {top, right, bottom},
            ],
        }
    }
}

impl From<u8> for PieceShape {
    fn from(x: u8) -> PieceShape {
        use PieceShape::*;
        match x {
            0 => I,
            1 => J,
            2 => L,
            3 => O,
            4 => S,
            5 => T,
            6 => Z,
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Piece {
    shape: PieceShape,
    rotation: Rotation,
    center: Position,
    color: Color,
}

impl Piece {
    fn sample(rng: &mut rand::rngs::SmallRng, palette: &[Color]) -> Piece {
        use rand::Rng;
        Piece {
            shape: PieceShape::from(rng.gen_range(0, PieceShape::COUNT)),
            rotation: Rotation::Verbatim,
            center: Position {
                x: Board::WIDTH as isize / 2,
                y: 2,
            },
            color: palette[rng.gen_range(0, palette.len())],
        }
    }

    fn ghost(&self, center: &Position) -> Piece {
        Piece {
            color: self.color.ghost(),
            center: *center,
            ..*self
        }
    }

    fn fields(&self) -> Vec<(Position, Field)> {
        self.shape
            .offsets()
            .iter()
            .zip(self.shape.borders().iter())
            .map(move |(offset, border)| {
                (
                    self.center + offset * &self.rotation,
                    Field {
                        border: border * &self.rotation,
                        color: self.color,
                    },
                )
            })
            .collect()
    }

    fn distance(&self, other: &Piece) -> usize {
        self.center.distance(&other.center) + self.rotation.distance(&other.rotation)
    }
}

impl std::ops::Add<Offset> for Piece {
    type Output = Piece;
    fn add(self, other: Offset) -> Piece {
        Piece {
            center: self.center + other,
            ..self
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Mul<Rotation> for Piece {
    type Output = Piece;
    fn mul(self, other: Rotation) -> Piece {
        Piece {
            rotation: self.rotation + other,
            ..self
        }
    }
}

impl std::ops::Add<Offset> for &Piece {
    type Output = Piece;
    fn add(self, other: Offset) -> Piece {
        Piece {
            center: self.center + other,
            ..*self
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl std::ops::Mul<Rotation> for &Piece {
    type Output = Piece;
    fn mul(self, other: Rotation) -> Piece {
        Piece {
            rotation: self.rotation + other,
            ..*self
        }
    }
}

#[derive(Debug)]
struct Board {
    rows: [[Option<Field>; Board::WIDTH]; Board::HEIGHT],
}

impl Board {
    const WIDTH: usize = 10;
    const HEIGHT: usize = 20;

    #[inline]
    fn fields(&self) -> impl Iterator<Item = (Position, Field)> + '_ {
        self.rows.iter().enumerate().flat_map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, cell)| {
                cell.and_then(|field| {
                    Some((
                        Position {
                            x: x as isize,
                            y: y as isize,
                        },
                        field,
                    ))
                })
            })
        })
    }

    #[inline]
    fn collides(&self, piece: &Piece) -> bool {
        piece
            .fields()
            .into_iter()
            .any(|(pos, _)| self.rows[pos.y as usize][pos.x as usize].is_some())
    }

    #[inline]
    fn supports(&self, piece: &Piece) -> bool {
        piece.fields().into_iter().any(|(pos, _)| {
            pos.y + 1 >= Board::HEIGHT as isize
                || self.rows[pos.y as usize + 1][pos.x as usize].is_some()
        })
    }

    #[inline]
    fn has_inside(&self, piece: &Piece) -> bool {
        piece.fields().into_iter().all(|(pos, _)| {
            (0..Board::HEIGHT as isize).contains(&pos.y)
                && (0..Board::WIDTH as isize).contains(&pos.x)
        })
    }

    #[inline]
    fn can_insert(&self, piece: &Piece) -> bool {
        self.has_inside(&piece) && !self.collides(&piece)
    }

    #[inline]
    fn try_insert(&mut self, piece: &Piece) -> bool {
        if !self.can_insert(piece) || !self.supports(piece) {
            return false;
        }
        piece
            .fields()
            .into_iter()
            .for_each(|(pos, f)| self.rows[pos.y as usize][pos.x as usize] = Some(f));
        true
    }

    #[inline]
    fn try_clear_one(&mut self) -> bool {
        for y in 0..Board::HEIGHT {
            if self.rows[y].iter().all(|f| f.is_some()) {
                for yy in (0..y).rev() {
                    self.rows[yy + 1] = self.rows[yy];
                }
                self.rows[0] = Default::default();
                return true;
            }
        }
        false
    }

    #[inline]
    fn try_clear(&mut self) -> usize {
        let mut n = 0;
        while self.try_clear_one() {
            n += 1;
        }
        n
    }
}

#[derive(Debug)]
struct CachedMesh {
    valid: bool,
    mesh: ggez::graphics::Mesh,
}

impl std::ops::Deref for CachedMesh {
    type Target = ggez::graphics::Mesh;

    fn deref(&self) -> &Self::Target {
        if !self.valid {
            panic!("dereferencing invalid cached mesh");
        }
        &self.mesh
    }
}

impl CachedMesh {
    fn empty(ctx: &mut Context) -> GameResult<CachedMesh> {
        Ok(CachedMesh {
            valid: false,
            mesh: ggez::graphics::Mesh::new_line(
                ctx,
                &[[0.0, 0.0], [1.0, 1.0]],
                1.0,
                ggez::graphics::WHITE,
            )?,
        })
    }
}

use std::collections::VecDeque;

type Move = either::Either<Offset, Rotation>;
type Solution = std::collections::VecDeque<Move>;

#[derive(Debug, Default)]
struct Pathfinder {
    subsol: [[[Option<Solution>; Rotation::COUNT]; Board::WIDTH]; Board::HEIGHT],
}

impl Pathfinder {
    fn is_exact_solution(
        &self,
        start: &Piece,
        endpos: &Position,
        endrot: &Rotation,
        sol: VecDeque<either::Either<Offset, Rotation>>,
    ) -> bool {
        let mut pos = start.center;
        let mut rot = start.rotation;
        for mov in sol {
            use either::Either::*;
            match mov {
                Left(movoff) => pos = pos + movoff,
                Right(movrot) => rot = rot + movrot,
            }
        }
        &pos == endpos && &rot == endrot
    }

    fn neighbour_moves(piece: &Piece) -> impl Iterator<Item = (Move, Piece)> {
        use either::Either::*;
        if piece.center.x + piece.center.y % 2 == 0 {
            vec![
                (Left(Offset { x: 1, y: 0 }), piece + Offset { x: 1, y: 0 }),
                (Left(Offset { x: 0, y: 1 }), piece + Offset { x: 0, y: 1 }),
                (Left(Offset { x: -1, y: 0 }), piece + Offset { x: -1, y: 0 }),
                (Left(Offset { x: 0, y: -1 }), piece + Offset { x: 0, y: -1 }),
                (
                    Right(Rotation::NinetyClockwise),
                    piece * Rotation::NinetyClockwise,
                ),
                (
                    Right(Rotation::NinetyCounterclockwise),
                    piece * Rotation::NinetyCounterclockwise,
                ),
            ]
            .into_iter()
        } else {
            vec![
                (Left(Offset { x: 0, y: 1 }), piece + Offset { x: 0, y: 1 }),
                (Left(Offset { x: 1, y: 0 }), piece + Offset { x: 1, y: 0 }),
                (Left(Offset { x: 0, y: -1 }), piece + Offset { x: 0, y: -1 }),
                (Left(Offset { x: -1, y: 0 }), piece + Offset { x: -1, y: 0 }),
                (
                    Right(Rotation::NinetyClockwise),
                    piece * Rotation::NinetyClockwise,
                ),
                (
                    Right(Rotation::NinetyCounterclockwise),
                    piece * Rotation::NinetyCounterclockwise,
                ),
            ]
            .into_iter()
        }
    }

    fn top_line(board: &Board) -> usize {
        board
            .rows
            .iter()
            .enumerate()
            .filter_map(|(i, row)| {
                if row.iter().any(|cell| cell.is_some()) {
                    Some(i)
                } else {
                    None
                }
            })
            .min()
            .unwrap_or(Board::HEIGHT)
    }

    fn count_board_holes(board: &Board, top: usize) -> usize {
        board
            .rows
            .iter()
            .enumerate()
            .skip(top)
            .map(|(i, row)| row.iter().filter(|f| f.is_none()).count() * (Board::HEIGHT - i))
            .sum::<usize>()
    }

    fn count_filled_holes(piece: &Piece, top: usize) -> usize {
        piece
            .fields()
            .iter()
            .filter(|(pos, _)| pos.y as usize >= top)
            .map(|(pos, _)| pos.y as usize)
            .sum::<usize>()
    }

    fn count_initiated_lines(piece: &Piece, top: usize) -> usize {
        top.saturating_sub(
            piece
                .fields()
                .iter()
                .filter(|(pos, _)| (pos.y as usize) < top)
                .map(|(pos, _)| Board::HEIGHT - pos.y as usize)
                .min()
                .unwrap_or(top),
        )
    }

    fn count_completed_lines(board: &Board, piece: &Piece, top: usize) -> usize {
        let ff = piece.fields();
        let ff: Vec<&Position> = ff.iter().map(|(pos, _)| pos).collect();
        board
            .rows
            .iter()
            .enumerate()
            .skip(top)
            .map(|(y, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(x, f)| {
                        f.is_none()
                            || ff
                                .iter()
                                .any(|pos| pos.y as usize == y && pos.x as usize == *x)
                    })
                    .count()
            })
            .filter(|c| *c == 0)
            .count()
    }

    const INITIATED_LINE_PENALTY: usize = 15;
    const BOARD_HOLE_PENALTY: usize = 1;
    const FILLED_HOLE_REWARD: usize = 8;
    const COMPLETED_LINE_REWARD: usize = 15;

    fn loss_debug(board: &Board, piece: &Piece) {
        let top = Pathfinder::top_line(board);
        let lines = Board::HEIGHT - top;
        let board_holes =
            Pathfinder::count_board_holes(board, top) * Pathfinder::BOARD_HOLE_PENALTY;
        let initiated_lines =
            Pathfinder::count_initiated_lines(piece, top) * Pathfinder::INITIATED_LINE_PENALTY;
        let filled_holes =
            Pathfinder::count_filled_holes(piece, top) * Pathfinder::FILLED_HOLE_REWARD;
        let completed_lines = Pathfinder::count_completed_lines(board, piece, top)
            * Pathfinder::COMPLETED_LINE_REWARD;
        dbg!(lines);
        dbg!(board_holes);
        dbg!(initiated_lines);
        dbg!(filled_holes);
        dbg!(completed_lines);
    }

    fn loss(board: &Board, piece: &Piece) -> isize {
        let top = Pathfinder::top_line(board);
        let lines = Board::HEIGHT - top;

        lines as isize
            + (Pathfinder::count_board_holes(board, top) * Pathfinder::BOARD_HOLE_PENALTY) as isize
            + (Pathfinder::count_initiated_lines(piece, top) * Pathfinder::INITIATED_LINE_PENALTY)
                as isize
            - (Pathfinder::count_filled_holes(piece, top) * Pathfinder::FILLED_HOLE_REWARD) as isize
            - (Pathfinder::count_completed_lines(board, piece, top)
                * Pathfinder::COMPLETED_LINE_REWARD) as isize
    }

    fn choose(board: &Board, start: &Piece, cumulative_loss: &mut isize) -> Piece {
        let mut pf = Pathfinder::default();
        let mut q = std::collections::VecDeque::new();
        let mut bestsol = *start;
        let mut bestloss = isize::MAX;
        q.push_back(*start);
        while let Some(piece) = q.pop_front() {
            if board.supports(&piece) {
                let new_loss = *cumulative_loss + Pathfinder::loss(board, &piece);
                if new_loss < bestloss {
                    bestloss = new_loss;
                    bestsol = piece;
                }
            }
            if pf.subsol[piece.center.y as usize][piece.center.x as usize]
                [usize::from(&piece.rotation)]
            .is_none()
            {
                q.extend(
                    Pathfinder::neighbour_moves(&piece)
                        .map(|(_, target)| target)
                        .filter(|target| board.can_insert(target))
                        .filter(|target| {
                            pf.subsol[target.center.y as usize][target.center.x as usize]
                                [usize::from(&target.rotation)]
                            .is_none()
                        }),
                );
                pf.subsol[piece.center.y as usize][piece.center.x as usize]
                    [usize::from(&piece.rotation)] = Some(Default::default());
            }
        }
        Pathfinder::loss_debug(board, &bestsol);
        dbg!(bestloss);
        *cumulative_loss = bestloss;
        bestsol
    }

    fn solve(board: &Board, start: &Piece, end: &Piece) -> Solution {
        let mut pf = Pathfinder::default();
        let mut q = std::collections::VecDeque::new();
        let mut bestsol = VecDeque::new();
        let mut bestdistance = usize::MAX;
        q.push_back((*start, Solution::new()));
        while let Some((piece, path)) = q.pop_front() {
            let new_dist = piece.distance(end);
            if new_dist < bestdistance && piece.rotation == end.rotation {
                bestdistance = new_dist;
                bestsol = path.clone();
            }
            if pf.subsol[piece.center.y as usize][piece.center.x as usize]
                [usize::from(&piece.rotation)]
            .is_none()
            {
                for (mov, target) in Pathfinder::neighbour_moves(&piece)
                    .filter(|(_, target)| board.can_insert(target))
                    .filter(|(_, target)| {
                        pf.subsol[target.center.y as usize][target.center.x as usize]
                            [usize::from(&target.rotation)]
                        .is_none()
                    })
                {
                    let mut new_path = path.clone();
                    new_path.push_back(mov);
                    q.push_back((target, new_path));
                }
                pf.subsol[piece.center.y as usize][piece.center.x as usize]
                    [usize::from(&piece.rotation)] = Some(path);
            }
            if bestdistance < 1 {
                break;
            }
        }
        bestsol
    }
}

#[derive(Debug)]
struct Game {
    palette: Palette,
    board: Board,
    rng: rand::rngs::SmallRng,
    current: Piece,
    ghost: Piece,
    next: Piece,
    show_grid: bool,
    mouse_enabled: bool,
    auto_mode: bool,
    auto_penalty: isize,
    board_mesh: CachedMesh,
    current_mesh: CachedMesh,
    ghost_mesh: CachedMesh,
    next_mesh: CachedMesh,
    grid_mesh: CachedMesh,
    scheduled_steps: Solution,
    scheduled_steps_valid: bool,
}

struct ScreenUtils {
    w: f32,
    h: f32,
    tile_size: f32,
    screen_offset: Offset,
}

impl Game {
    fn new(ctx: &mut ggez::Context) -> GameResult<Game> {
        use rand::SeedableRng;
        let palette = Palette::default();
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let current = Piece::sample(&mut rng, &palette.pieces);
        let board = Board {
            rows: Default::default(),
        };
        Ok(Game {
            board,
            current,
            ghost: current.ghost(&current.center),
            next: Piece::sample(&mut rng, &palette.pieces),
            show_grid: false,
            mouse_enabled: true,
            auto_mode: false,
            auto_penalty: -50,
            rng,
            palette,
            board_mesh: CachedMesh::empty(ctx)?,
            current_mesh: CachedMesh::empty(ctx)?,
            ghost_mesh: CachedMesh::empty(ctx)?,
            grid_mesh: CachedMesh::empty(ctx)?,
            next_mesh: CachedMesh::empty(ctx)?,
            scheduled_steps: Solution::new(),
            scheduled_steps_valid: true,
        })
    }

    fn prepare_auto(&mut self) {
        let new_target = Pathfinder::choose(&self.board, &self.current, &mut self.auto_penalty);
        self.target_move_to(new_target.center);
        self.target_set_rotation(new_target.rotation);
    }

    fn target_invalidate(&mut self) {
        self.ghost_mesh.valid = false;
        self.scheduled_steps_valid = false;
    }

    fn target_rotate(&mut self, diff: Rotation) {
        self.ghost.rotation = self.ghost.rotation + diff;
        self.target_invalidate();
    }

    fn target_set_rotation(&mut self, rot: Rotation) {
        self.ghost.rotation = rot;
        self.target_invalidate();
    }

    fn target_move_to(&mut self, pos: Position) {
        self.ghost.center = pos;
        self.target_invalidate();
    }

    fn target_move(&mut self, off: Offset) {
        self.ghost.center = self.ghost.center.saturating_add(off);
        self.target_invalidate();
    }

    fn exec_rotate(&mut self, diff: Rotation) {
        if diff == Rotation::UpsideDown {
            panic!("attempt to perform queued rotation by more than one");
        }
        let new_piece = self.current * diff;
        if !self.board.can_insert(&new_piece) {
            panic!("attempt to perform queued rotation to invalid target");
        }
        self.current = new_piece;
        self.current_mesh.valid = false;
    }

    fn exec_move(&mut self, diff: Offset) {
        if diff.x.abs() + diff.y.abs() > 1 {
            panic!("attempt to perform queued move by more than one");
        }
        let new_piece = self.current + diff;
        if !self.board.can_insert(&new_piece) {
            panic!("attempt to perform queued move to invalid target");
        }
        self.current = new_piece;
        self.current_mesh.valid = false;
    }

    fn try_place(&mut self, ctx: &mut Context) {
        if self.board.try_insert(&self.current) {
            self.board_mesh.valid = false;
            if !self.board.can_insert(&self.next) {
                if self.auto_mode {
                    *self = Game {
                        show_grid: self.show_grid,
                        mouse_enabled: self.mouse_enabled,
                        auto_mode: self.auto_mode,
                        ..Game::new(ctx).unwrap()
                    };
                } else {
                    // TODO
                }
            }
            self.current = self.next;
            self.current_mesh.valid = false;
            self.ghost = self.current.ghost(&self.ghost.center);
            self.target_invalidate();
            self.next = Piece::sample(&mut self.rng, &self.palette.pieces);
            self.next_mesh.valid = false;
            self.board.try_clear();
            if self.auto_mode {
                self.prepare_auto();
            }
        }
    }

    fn screen_utils(&self, ctx: &Context) -> ScreenUtils {
        let (w, h) = ggez::graphics::size(ctx);
        let tile_size = (w / (1 + Board::WIDTH + 7) as f32).min(h / (1 + Board::HEIGHT + 1) as f32);
        let screen_offset = if (7 + Board::WIDTH + 7) as f32 * tile_size <= w {
            Offset {
                x: (w - tile_size * Board::WIDTH as f32) as isize / 2,
                y: (h - tile_size * Board::HEIGHT as f32) as isize / 2,
            }
        } else {
            Offset {
                x: (w - tile_size * (Board::WIDTH + 6) as f32) as isize / 2,
                y: (h - tile_size * Board::HEIGHT as f32) as isize / 2,
            }
        };
        ScreenUtils {
            w,
            h,
            tile_size,
            screen_offset,
        }
    }
}

impl ggez::event::EventHandler for Game {
    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        use ggez::graphics::{self, Drawable};
        let ScreenUtils {
            w,
            h,
            tile_size,
            screen_offset,
        } = self.screen_utils(ctx);

        let needs_redraw = !(self.board_mesh.valid
            && self.current_mesh.valid
            && self.ghost_mesh.valid
            && self.grid_mesh.valid
            && self.next_mesh.valid);

        if !self.board_mesh.valid {
            let mut mesh = &mut ggez::graphics::MeshBuilder::new();

            // clear background (needed on GLES)
            mesh = mesh.rectangle(
                graphics::DrawMode::fill(),
                graphics::Rect {
                    x: 0.0,
                    y: 0.0,
                    w,
                    h,
                },
                self.palette.bg,
            );

            // draw board border
            mesh = mesh.rectangle(
                graphics::DrawMode::stroke(tile_size / 10.0),
                graphics::Rect {
                    x: screen_offset.x as f32,
                    y: screen_offset.y as f32,
                    w: tile_size * Board::WIDTH as f32,
                    h: tile_size * Board::HEIGHT as f32,
                },
                graphics::BLACK,
            );

            // draw board background
            mesh = mesh.rectangle(
                graphics::DrawMode::fill(),
                graphics::Rect {
                    x: screen_offset.x as f32,
                    y: screen_offset.y as f32,
                    w: tile_size * Board::WIDTH as f32,
                    h: tile_size * Board::HEIGHT as f32,
                },
                self.palette.board,
            );

            // draw fields
            for (pos, field) in self.board.fields() {
                mesh = field.mesh(mesh, pos, tile_size, screen_offset);
            }

            self.board_mesh = CachedMesh {
                valid: true,
                mesh: mesh.build(ctx)?,
            };
        }

        if !self.current_mesh.valid {
            self.current_mesh = CachedMesh {
                valid: true,
                mesh: self
                    .current
                    .fields()
                    .into_iter()
                    .fold(
                        &mut ggez::graphics::MeshBuilder::new(),
                        |mb, (pos, field)| field.mesh(mb, pos, tile_size, screen_offset),
                    )
                    .build(ctx)?,
            };
        }

        if !self.ghost_mesh.valid {
            self.ghost_mesh = CachedMesh {
                valid: true,
                mesh: self
                    .ghost
                    .fields()
                    .into_iter()
                    .fold(
                        &mut ggez::graphics::MeshBuilder::new(),
                        |mb, (pos, field)| field.mesh(mb, pos, tile_size, screen_offset),
                    )
                    .build(ctx)?,
            };
        }

        if !self.next_mesh.valid {
            let mut mesh = ggez::graphics::MeshBuilder::new();
            let next_rect = graphics::Rect {
                x: screen_offset.x as f32 + tile_size * (Board::WIDTH + 1) as f32,
                y: screen_offset.y as f32,
                w: tile_size * 5f32,
                h: tile_size * 5f32,
            };
            mesh = mesh
                .rectangle(graphics::DrawMode::fill(), next_rect, self.palette.board)
                .rectangle(
                    graphics::DrawMode::stroke(tile_size / 10.0),
                    next_rect,
                    self.palette.board.darken(),
                )
                .clone();
            self.next_mesh = CachedMesh {
                valid: true,
                mesh: self
                    .next
                    .fields()
                    .into_iter()
                    .fold(&mut mesh, |mb, (pos, field)| {
                        field.mesh(
                            mb,
                            pos + Offset {
                                x: (Board::WIDTH as isize - self.next.center.x + 3) as isize,
                                y: 0,
                            },
                            tile_size,
                            screen_offset,
                        )
                    })
                    .build(ctx)?,
            };
        }

        if self.show_grid && !self.grid_mesh.valid {
            self.grid_mesh = CachedMesh {
                valid: true,
                mesh: (1..Board::WIDTH)
                    .fold(
                        (1..Board::HEIGHT).fold(
                            &mut ggez::graphics::MeshBuilder::new(),
                            |mb, y| {
                                mb.line(
                                    &[
                                        na::Point2::new(
                                            screen_offset.x as f32,
                                            screen_offset.y as f32 + tile_size * y as f32,
                                        ),
                                        na::Point2::new(
                                            screen_offset.x as f32
                                                + tile_size * Board::WIDTH as f32,
                                            screen_offset.y as f32 + tile_size * y as f32,
                                        ),
                                    ],
                                    tile_size / 10f32,
                                    self.palette.grid,
                                )
                                .unwrap()
                            },
                        ),
                        |mb, x| {
                            mb.line(
                                &[
                                    na::Point2::new(
                                        screen_offset.x as f32 + tile_size * x as f32,
                                        screen_offset.y as f32,
                                    ),
                                    na::Point2::new(
                                        screen_offset.x as f32 + tile_size * x as f32,
                                        screen_offset.y as f32 + tile_size * Board::HEIGHT as f32,
                                    ),
                                ],
                                tile_size / 10f32,
                                self.palette.grid,
                            )
                            .unwrap()
                        },
                    )
                    .build(ctx)?,
            };
        }

        if needs_redraw {
            graphics::clear(ctx, self.palette.bg);
            (*self.board_mesh).draw(ctx, graphics::DrawParam::default())?;
            (*self.current_mesh).draw(ctx, graphics::DrawParam::default())?;
            (*self.ghost_mesh).draw(ctx, graphics::DrawParam::default())?;
            (*self.next_mesh).draw(ctx, graphics::DrawParam::default())?;
            if self.show_grid {
                (*self.grid_mesh).draw(ctx, graphics::DrawParam::default())?;
            }
            graphics::present(ctx)?;
        }
        ggez::timer::yield_now();
        Ok(())
    }

    fn resize_event(&mut self, ctx: &mut Context, width: f32, height: f32) {
        self.board_mesh.valid = false;
        self.current_mesh.valid = false;
        self.ghost_mesh.valid = false;
        self.next_mesh.valid = false;
        self.grid_mesh.valid = false;
        ggez::graphics::set_screen_coordinates(
            ctx,
            ggez::graphics::Rect {
                x: 0f32,
                y: 0f32,
                w: width,
                h: height,
            },
        )
        .unwrap();
    }

    fn mouse_wheel_event(&mut self, _ctx: &mut Context, _x: f32, y: f32) {
        if self.auto_mode || !self.mouse_enabled {
            return;
        }
        let y = -y;
        if y >= 1f32 {
            self.target_rotate((y.trunc() as usize).into());
        }
        if y <= 1f32 {
            self.target_rotate(-Rotation::from(-y.trunc() as usize));
        }
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        keycode: ggez::input::keyboard::KeyCode,
        keymods: ggez::input::keyboard::KeyMods,
        _repeat: bool,
    ) {
        use ggez::input::keyboard::KeyCode;
        if self.auto_mode {
            return;
        }
        if keymods.is_empty() {
            match keycode {
                KeyCode::Up | KeyCode::W => self.target_move(Offset { x: 0, y: -1 }),
                KeyCode::Down | KeyCode::S => self.target_move(Offset { x: 0, y: 1 }),
                KeyCode::Left | KeyCode::A => self.target_move(Offset { x: -1, y: 0 }),
                KeyCode::Right | KeyCode::D => self.target_move(Offset { x: 1, y: 0 }),
                KeyCode::LBracket | KeyCode::Q => {
                    self.target_rotate(Rotation::NinetyCounterclockwise)
                }
                KeyCode::RBracket | KeyCode::E => self.target_rotate(Rotation::NinetyClockwise),
                _ => (),
            }
        }
    }

    fn key_up_event(
        &mut self,
        ctx: &mut Context,
        keycode: ggez::input::keyboard::KeyCode,
        keymods: ggez::input::keyboard::KeyMods,
    ) {
        use ggez::input::keyboard::KeyCode;
        if keymods.is_empty() {
            match keycode {
                KeyCode::G => {
                    self.show_grid = !self.show_grid;
                }
                KeyCode::M => {
                    self.mouse_enabled = !self.mouse_enabled;
                }
                KeyCode::O => {
                    self.auto_mode = !self.auto_mode;
                    if self.auto_mode {
                        self.prepare_auto();
                    }
                }
                KeyCode::Space | KeyCode::Return if !self.auto_mode => self.try_place(ctx),
                KeyCode::Escape => ggez::event::quit(ctx),
                _ => {}
            }
        }
    }

    fn mouse_motion_event(&mut self, ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        if self.auto_mode || !self.mouse_enabled {
            return;
        }
        let ScreenUtils {
            w: _,
            h: _,
            tile_size,
            screen_offset,
        } = self.screen_utils(ctx);
        let x = (((x - screen_offset.x as f32) / tile_size) as isize)
            .max(0)
            .min(Board::WIDTH as isize - 1);
        let y = (((y - screen_offset.y as f32) / tile_size) as isize)
            .max(0)
            .min(Board::HEIGHT as isize - 1);
        self.target_move_to(Position { x, y });
    }

    fn mouse_button_up_event(
        &mut self,
        ctx: &mut Context,
        button: ggez::input::mouse::MouseButton,
        _x: f32,
        _y: f32,
    ) {
        if self.auto_mode || !self.mouse_enabled {
            return;
        }
        match button {
            ggez::input::mouse::MouseButton::Left => {
                self.try_place(ctx);
            }
            ggez::input::mouse::MouseButton::Right => {
                self.ghost.rotation = self.ghost.rotation + 1.into();
                self.ghost_mesh.valid = false;
            }
            _ => {}
        }
    }

    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        use either::Either::*;
        if !self.scheduled_steps_valid {
            self.scheduled_steps = Pathfinder::solve(&self.board, &self.current, &self.ghost);
            self.scheduled_steps_valid = true;
        }
        match self.scheduled_steps.pop_front() {
            Some(Left(off)) => self.exec_move(off),
            Some(Right(rot)) => self.exec_rotate(rot),
            None => (),
        }
        if self.auto_mode && self.scheduled_steps.is_empty() {
            self.try_place(ctx);
        }
        Ok(())
    }
}
