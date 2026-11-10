//! Private module for selective re-export.

use crate::checker::{Checker, EventuallyBits, Expectation, Path};
use crate::job_market::JobBroker;
use crate::{CheckerBuilder, CheckerVisitor, Fingerprint, Model, Property, fingerprint};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use nohash_hasher::NoHashHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hash};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::SystemTime;

// While this file is currently quite similar to dfs.rs, a refactoring to lift shared
// behavior is being postponed until DPOR is implemented.

pub(crate) struct BfsChecker<M: Model> {
    // Immutable state.
    model: Arc<M>,
    handles: Vec<std::thread::JoinHandle<()>>,

    // Mutable state.
    job_broker: JobBroker<Job<M::State>>,
    state_count: Arc<AtomicUsize>,
    max_depth: Arc<AtomicUsize>,
    /// The depth of the shortest path found so far to each generated state.
    generated: Arc<DashMap<Fingerprint, NonZeroUsize, BuildHasherDefault<NoHashHasher<u64>>>>,
    // In the original fingerprint-based Path construction, the value type of `discovery` was a
    // single fingerprint since the path could be reconstructed in reverse using the parent
    // relationships in `generated`. However, with action indices encoding, this is not possible,
    // so we must store a Vec (like in dfs.rs).
    discoveries: Arc<DashMap<&'static str, Vec<usize>>>,
}
type Job<State> = (State, EventuallyBits, NonZeroUsize, Vec<usize>);

impl<M> BfsChecker<M>
where
    M: Model + Send + Sync + 'static,
    M::State: Hash + Send + 'static,
{
    pub(crate) fn spawn(options: CheckerBuilder<M>) -> Self
    where
        M::State: Clone + PartialEq,
        M::Action: Clone + PartialEq,
    {
        let model = Arc::new(options.model);
        let target_state_count = options.target_state_count;
        let target_max_depth = options.target_max_depth;
        let thread_count = options.thread_count;
        let visitor = Arc::new(options.visitor);
        let finish_when = Arc::new(options.finish_when);
        let properties = Arc::new(model.properties());

        let init_states: Vec<_> = model
            .init_states()
            .into_iter()
            .filter(|s| model.within_boundary(s))
            .collect();
        let state_count = Arc::new(AtomicUsize::new(init_states.len()));
        let max_depth = Arc::new(AtomicUsize::new(0));
        let generated = Arc::new({
            let generated = DashMap::default();
            for s in &init_states {
                generated.insert(fingerprint(s), NonZeroUsize::new(1).unwrap());
            }
            generated
        });
        let ebits = {
            let mut ebits = EventuallyBits::new();
            for (i, p) in model.properties().iter().enumerate() {
                if let Property {
                    expectation: Expectation::Eventually,
                    ..
                } = p
                {
                    ebits.insert(i);
                }
            }
            ebits
        };
        let pending: VecDeque<_> = init_states
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, ebits.clone(), NonZeroUsize::new(1).unwrap(), vec![i]))
            .collect();
        let discoveries = Arc::new(DashMap::default());
        let mut handles = Vec::new();

        let close_at = options.timeout.map(|t| SystemTime::now() + t);
        let mut job_broker = JobBroker::new(thread_count, close_at);
        job_broker.push(pending);

        for t in 0..thread_count {
            let model = Arc::clone(&model);
            let visitor = Arc::clone(&visitor);
            let finish_when = Arc::clone(&finish_when);
            let properties = Arc::clone(&properties);
            let mut job_broker = job_broker.clone();
            let state_count = Arc::clone(&state_count);
            let max_depth = Arc::clone(&max_depth);
            let generated = Arc::clone(&generated);
            let discoveries = Arc::clone(&discoveries);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("checker-{t}"))
                    .spawn(move || {
                        log::debug!("{t}: Thread started.");
                        let mut pending = VecDeque::new();
                        loop {
                            // Step 1: Do work.
                            if pending.is_empty() {
                                pending = {
                                    let jobs = job_broker.pop();
                                    if jobs.is_empty() {
                                        log::debug!(
                                            "{}: No more work. Shutting down... gen={}",
                                            t,
                                            generated.len()
                                        );
                                        return;
                                    }
                                    log::trace!("{}: Job found. size={}", t, jobs.len());
                                    jobs
                                };
                            }
                            Self::check_block(
                                &model,
                                &state_count,
                                &generated,
                                &mut pending,
                                &discoveries,
                                &visitor,
                                1500,
                                target_max_depth,
                                &max_depth,
                            );
                            if finish_when.matches(
                                &discoveries.iter().map(|r| *r.key()).collect(),
                                &properties,
                            ) {
                                log::debug!(
                                    "{}: Discovery complete. Shutting down... gen={}",
                                    t,
                                    generated.len()
                                );
                                return;
                            }
                            if let Some(target_state_count) = target_state_count
                                && target_state_count.get() <= state_count.load(Ordering::Relaxed)
                            {
                                log::debug!(
                                    "{}: Reached target state count. Shutting down... gen={}",
                                    t,
                                    generated.len()
                                );
                                return;
                            }

                            // Step 2: Share work.
                            if pending.len() > 1 && thread_count > 1 {
                                job_broker.split_and_push(&mut pending);
                            }
                        }
                    })
                    .expect("Failed to spawn a thread"),
            );
        }
        BfsChecker {
            model,
            handles,
            job_broker,
            state_count,
            max_depth,
            generated,
            discoveries,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_block(
        model: &M,
        state_count: &AtomicUsize,
        generated: &DashMap<Fingerprint, NonZeroUsize, BuildHasherDefault<NoHashHasher<u64>>>,
        pending: &mut VecDeque<Job<M::State>>,
        discoveries: &DashMap<&'static str, Vec<usize>>,
        visitor: &Option<Box<dyn CheckerVisitor<M> + Send + Sync>>,
        mut max_count: usize,
        target_max_depth: Option<NonZeroUsize>,
        global_max_depth: &AtomicUsize,
    ) where
        M::State: Clone + PartialEq,
        M::Action: Clone + PartialEq,
    {
        let properties = model.properties();

        let mut current_max_depth = global_max_depth.load(Ordering::Relaxed);
        let mut actions = Vec::new();
        loop {
            // Done if reached max count.
            if max_count == 0 {
                return;
            }
            max_count -= 1;

            // Done if none pending.
            let (state, mut ebits, max_depth, action_path) = match pending.pop_back() {
                None => return,
                Some(pair) => pair,
            };

            if max_depth.get() > current_max_depth {
                let _ = global_max_depth.compare_exchange(
                    current_max_depth,
                    max_depth.get(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                current_max_depth = max_depth.get();
            }

            if let Some(visitor) = visitor {
                visitor.visit(
                    model,
                    Path::from_action_indices(model, VecDeque::from(action_path.clone())),
                );
            }

            // Done if discoveries found for all properties.
            let mut is_awaiting_discoveries = false;
            for (i, property) in properties.iter().enumerate() {
                // The cheap tests come first: the discoveries map is shared
                // between threads, and most properties are not eventualities.
                if property.expectation == Expectation::Eventually
                    && !ebits.contains(i)
                    && discoveries.contains_key(property.name)
                {
                    continue;
                }
                match property {
                    Property {
                        expectation: Expectation::Always,
                        condition: always,
                        ..
                    } => {
                        if !always(model, &state) {
                            record_discovery(discoveries, property.name, &action_path);
                        } else {
                            is_awaiting_discoveries = true;
                        }
                    }
                    Property {
                        expectation: Expectation::Sometimes,
                        condition: sometimes,
                        ..
                    } => {
                        if sometimes(model, &state) {
                            record_discovery(discoveries, property.name, &action_path);
                        } else {
                            is_awaiting_discoveries = true;
                        }
                    }
                    Property {
                        expectation: Expectation::Eventually,
                        condition: eventually,
                        ..
                    } => {
                        // The checker early exits after finding discoveries for every property,
                        // and "eventually" property discoveries are only identified at terminal
                        // states, so if we are here it means we are still awaiting a corresponding
                        // discovery regardless of whether the eventually property is now satisfied
                        // (i.e. it might be falsifiable via a different path).
                        is_awaiting_discoveries = true;
                        if eventually(model, &state) {
                            ebits.remove(i);
                        }
                    }
                }
            }
            if !is_awaiting_discoveries {
                return;
            }

            // States at the depth limit are checked above but not expanded.
            if let Some(target_max_depth) = target_max_depth
                && max_depth >= target_max_depth
            {
                log::trace!("Not expanding state at max depth {max_depth}");
                continue;
            }

            // Otherwise enqueue newly generated states (with related metadata).
            let mut is_terminal = true;
            model.actions(&state, &mut actions);
            for (action_idx, action) in actions.drain(..).enumerate() {
                let next_state = model.next_state(&state, action);
                if next_state.is_none() {
                    continue;
                }
                let next_state = next_state.unwrap();
                // Skip if outside boundary.
                if !model.within_boundary(&next_state) {
                    continue;
                }
                state_count.fetch_add(1, Ordering::Relaxed);

                // Skip if already generated.
                //
                // FIXME: we should really include ebits in the fingerprint here --
                // it is possible to arrive at a DAG join with two different ebits
                // values, and subsequently treat the fact that some eventually
                // property held on the path leading to the first visit as meaning
                // that it holds in the path leading to the second visit -- another
                // possible false-negative.
                let next_fingerprint = fingerprint(&next_state);
                let next_depth = NonZeroUsize::new(max_depth.get() + 1).unwrap();
                match generated.entry(next_fingerprint) {
                    Entry::Vacant(next_entry) => {
                        next_entry.insert(next_depth);
                    }
                    // Threads do not move through the levels in lockstep, so a
                    // state can first arrive by a path longer than its shortest
                    // one. Under a depth limit that would cut off states within
                    // the limit, so a shorter arrival expands the state again.
                    Entry::Occupied(mut next_entry)
                        if target_max_depth.is_some() && next_depth < *next_entry.get() =>
                    {
                        next_entry.insert(next_depth);
                    }
                    _ => {
                        // FIXME: arriving at an already-known state may be a loop (in which case it
                        // could, in a fancier implementation, be considered a terminal state for
                        // purposes of eventually-property checking) but it might also be a join in
                        // a DAG, which makes it non-terminal. These cases can be disambiguated (at
                        // some cost), but for now we just _don't_ treat them as terminal, and tell
                        // users they need to explicitly ensure model path-acyclicality when they're
                        // using eventually properties (using a boundary or empty actions or
                        // whatever).
                        is_terminal = false;
                        continue;
                    }
                }

                // Otherwise further checking is applicable. Only a state that
                // will be visited needs its path, so the path is built here,
                // after the duplicates are gone, and in one allocation.
                is_terminal = false;
                let mut next_action_path = Vec::with_capacity(action_path.len() + 1);
                next_action_path.extend_from_slice(&action_path);
                next_action_path.push(action_idx);
                pending.push_front((next_state, ebits.clone(), next_depth, next_action_path));
            }
            if is_terminal {
                for (i, property) in properties.iter().enumerate() {
                    if ebits.contains(i) {
                        discoveries.insert(property.name, action_path.clone());
                    }
                }
            }
        }
    }
}

/// Records an `always` or `sometimes` discovery unless one with a path at most
/// as long is already recorded, so the run reports the shortest path it found.
fn record_discovery(
    discoveries: &DashMap<&'static str, Vec<usize>>,
    name: &'static str,
    path: &[usize],
) {
    match discoveries.entry(name) {
        Entry::Vacant(entry) => {
            entry.insert(path.to_vec());
        }
        Entry::Occupied(mut entry) => {
            if path.len() < entry.get().len() {
                entry.insert(path.to_vec());
            }
        }
    }
}

impl<M> Checker<M> for BfsChecker<M>
where
    M: Model,
    M::State: Hash,
{
    fn model(&self) -> &M {
        &self.model
    }

    fn state_count(&self) -> usize {
        self.state_count.load(Ordering::Relaxed)
    }

    fn unique_state_count(&self) -> usize {
        self.generated.len()
    }

    fn max_depth(&self) -> usize {
        self.max_depth.load(Ordering::Relaxed)
    }

    fn discoveries(&self) -> HashMap<&'static str, Path<M::State, M::Action>>
    where
        M::State: Clone + PartialEq,
        M::Action: Clone + PartialEq,
    {
        self.discoveries
            .iter()
            .map(|mapref| {
                (
                    *mapref.key(),
                    Path::from_action_indices(self.model(), VecDeque::from(mapref.value().clone())),
                )
            })
            .collect()
    }

    fn handles(&mut self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut self.handles)
    }

    fn is_done(&self) -> bool {
        self.job_broker.is_closed() || self.discoveries.len() == self.model.properties().len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::linear_equation_solver::*;
    use crate::*;

    #[test]
    fn visits_states_in_bfs_order() {
        let (recorder, accessor) = StateRecorder::new_with_accessor();
        LinearEquation { a: 2, b: 10, c: 14 }
            .checker()
            .visitor(recorder)
            .spawn_bfs()
            .join();
        assert_eq!(
            accessor(),
            vec![
                // distance == 0
                (0, 0),
                // distance == 1
                (1, 0),
                (0, 1),
                // distance == 2
                (2, 0),
                (1, 1),
                (0, 2),
                // distance == 3
                (3, 0),
                (2, 1),
            ]
        );
    }

    #[test]
    fn can_complete_by_enumerating_all_states() {
        let checker = LinearEquation { a: 2, b: 4, c: 7 }
            .checker()
            .spawn_bfs()
            .join();
        assert!(checker.is_done());
        checker.assert_no_discovery("solvable");
        assert_eq!(checker.unique_state_count(), 256 * 256);
    }

    #[test]
    fn can_complete_by_eliminating_properties() {
        let checker = LinearEquation { a: 2, b: 10, c: 14 }
            .checker()
            .spawn_bfs()
            .join();
        checker.assert_properties();
        assert_eq!(checker.unique_state_count(), 12);

        // bfs found this example...
        assert_eq!(
            checker.discovery("solvable").unwrap().into_actions(),
            // (2*2 + 10*1) % 256 == 14
            vec![Guess::IncreaseX, Guess::IncreaseX, Guess::IncreaseY,]
        );
        // ... but there are of course other solutions, such as the following.
        checker.assert_discovery(
            "solvable",
            // (2*0 + 10*27) % 256 == 14
            vec![Guess::IncreaseY; 27],
        );
    }

    /// Counts up from zero, by one or by two, in three independent counters.
    /// Most states are reachable by paths of several lengths.
    /// The flag turns on a property that fails once the first counter reaches 2.
    struct Steps(bool);

    impl Model for Steps {
        type State = [u8; 3];
        type Action = (usize, u8);

        fn init_states(&self) -> Vec<Self::State> {
            vec![[0, 0, 0]]
        }

        fn actions(&self, _state: &Self::State, actions: &mut Vec<Self::Action>) {
            for i in 0..3 {
                actions.push((i, 1));
                actions.push((i, 2));
            }
        }

        fn next_state(&self, state: &Self::State, (i, n): Self::Action) -> Option<Self::State> {
            let mut next = *state;
            next[i] = next[i].checked_add(n)?;
            Some(next)
        }

        fn properties(&self) -> Vec<Property<Self>> {
            if self.0 {
                vec![Property::always("below 2", |_, s: &[u8; 3]| s[0] < 2)]
            } else {
                // The checker stops once every property has a discovery, so
                // a property that never has one makes it visit every state.
                vec![Property::sometimes("unreachable", |_, _| false)]
            }
        }
    }

    #[test]
    fn checks_properties_of_states_at_the_depth_limit() {
        // [2, 0, 0] is one action away from the initial state, so a depth limit
        // of 2 generates it, and the checker must check it.
        let checker = Steps(true).checker().target_max_depth(2).spawn_bfs().join();
        assert_eq!(checker.unique_state_count(), 7);
        assert!(checker.discovery("below 2").is_some());
    }

    #[test]
    fn depth_limit_is_exact_with_many_threads() {
        // Every state within 20 actions of the start. A counter at value v
        // needs ceil(v / 2) actions on its shortest path.
        let mut expected = 0;
        for a in 0u32..=40 {
            for b in 0u32..=40 {
                for c in 0u32..=40 {
                    if a.div_ceil(2) + b.div_ceil(2) + c.div_ceil(2) <= 20 {
                        expected += 1;
                    }
                }
            }
        }
        for _ in 0..20 {
            let checker = Steps(false)
                .checker()
                .target_max_depth(21)
                .threads(8)
                .spawn_bfs()
                .join();
            assert_eq!(checker.unique_state_count(), expected);
        }
    }

    // test that the checker shuts down all threads properly after a checker thread encounters a
    // panic in the model execution.
    #[test]
    #[should_panic]
    fn handles_panics_gracefully() {
        crate::test_util::panicker::Panicker
            .checker()
            .threads(2)
            .spawn_bfs()
            .join();
    }
}
