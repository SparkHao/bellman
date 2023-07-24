use std::sync::atomic::{AtomicUsize, Ordering};
use std::env;
use log::{info, debug};
use std::thread::sleep;
use std::time;
use rand::Rng;

pub const WAIT_GPU: u64 = 1000;
pub const WAIT_FACT_SEAL: u64 = 10;

lazy_static::lazy_static! {
    pub static ref ALL_GPUS: Vec<AtomicUsize> = init_all_gpus((GPU_LIST()).len(), GPU_MEM());
}

fn get_numeric_env(env_key: String, def: usize) -> usize {
    let res = if let Ok(num) = env::var(env_key) {
        if let Ok(num) = num.parse() {
            num
        } else {
            def
        }
    } else {
        def
    };
    res
}

pub fn get_env_swicher(name: String, def: bool) -> bool {
    let res = if let Ok(switch) = env::var(name.clone()) {
        if switch == "true" {
            true
        } else if switch == "false" {
            false
        }
        else {
            def
        }
    } else {
        def
    };
    debug!("The switch of {:?} is set to {:?}", name, res);
    res
}

pub fn min_index(array: &[usize]) -> (usize, usize) {
    let mut min_gpu_idx = 0;
    let mut min_gpu_id = (GPU_LIST())[0];
    let mut min_gpu_mem = array[0];

    for i in 1..(GPU_LIST()).len() {
        if min_gpu_mem>array[i] {
            min_gpu_idx = i;
            min_gpu_id = (GPU_LIST())[i];
            min_gpu_mem = array[i];
        }
    }

    (min_gpu_idx, min_gpu_id)
}

pub fn init_all_gpus(gpu_num: usize, _gpu_mem: usize) -> Vec<AtomicUsize> {
    let mut all_gpus= Vec::new();
    for _i in 0..gpu_num {
        all_gpus.push(AtomicUsize::new(181));
    }
    all_gpus
}

pub fn get_all_gpus_consumed_mem() -> Vec<usize> {
    let mut all_gpus_consumed = Vec::new();
    for i in 0..(GPU_LIST()).len() {
        all_gpus_consumed.push((*ALL_GPUS)[i].load(Ordering::SeqCst));
    }
    all_gpus_consumed
}

pub fn free_gpu_ok(consume_mem: usize) -> (bool, usize, usize) {
    let all_gpus_consumed = get_all_gpus_consumed_mem();
    debug!("All GPU consumed memory: {:?}", &all_gpus_consumed);
    let (min_idx, gpu_id) = min_index(&all_gpus_consumed);
    if (*ALL_GPUS)[min_idx].load(Ordering::SeqCst) < (GPU_MEM())-consume_mem {
        let new_used_mem = (*ALL_GPUS)[min_idx].fetch_add(consume_mem, Ordering::SeqCst);
        info!("device idx {:}, gpu_id: {} used gpu memory: {:?}", min_idx, gpu_id, new_used_mem+consume_mem);
        (true, min_idx, gpu_id)
    } else {
        (false, min_idx, gpu_id)
    }
}

pub fn gpu_overloaded(min_idx: usize) -> bool {
    (*ALL_GPUS)[min_idx].load(Ordering::SeqCst) >= (GPU_MEM())
}

pub fn finish_use_gpu(min_idx: usize, consume_mem: usize) {
    (*ALL_GPUS)[min_idx].fetch_sub(consume_mem, Ordering::SeqCst);
}

pub fn cpu_start() {
    CPU_RNNING_NUM.fetch_add(1, Ordering::SeqCst);
    debug!("Add one cpu task thread!");
}

pub fn cpu_finish() {
    CPU_RNNING_NUM.fetch_sub(1, Ordering::SeqCst);
    debug!("Sub one cpu task thread!");
}

pub fn cpu_running() -> bool {
    debug!("Running CPU tasks thread: {:?}", CPU_RNNING_NUM.load(Ordering::SeqCst));
    CPU_RNNING_NUM.load(Ordering::SeqCst) >= MIN_CPU_NUM()
}


pub fn hash_start() {
    HASH_RNNING_NUM.fetch_add(1, Ordering::SeqCst);
}

pub fn hash_finish() {
    HASH_RNNING_NUM.fetch_sub(1, Ordering::SeqCst);
}

pub fn hash_rnning() -> bool {
    if HASH_FIRST() != 0 {
        debug!("HASH_FIRST set to none zero, Running hash thread: {:?}", HASH_RNNING_NUM.load(Ordering::SeqCst));
        HASH_RNNING_NUM.load(Ordering::SeqCst) >= HASH_FIRST()
    }
    else {
        debug!("HASH_FIRST set to false");
        false
    }
}

pub fn bell_start() {
    BELL_RNNING_NUM.fetch_add(1, Ordering::SeqCst);
}

pub fn bell_finish() {
    BELL_RNNING_NUM.fetch_sub(1, Ordering::SeqCst);
}

pub fn get_waiting_bell() -> usize {
    BELL_RNNING_NUM.load(Ordering::SeqCst)
}


static CPU_RNNING_NUM: AtomicUsize = AtomicUsize::new(0);
static HASH_RNNING_NUM: AtomicUsize = AtomicUsize::new(0);
static BELL_RNNING_NUM: AtomicUsize = AtomicUsize::new(0);

pub fn should_opt_cpu() -> bool {
    let mut rng = rand::thread_rng();
    let rand_sleep = rng.gen_range(0..100);
    debug!("Random sleep {:?} MS", rand_sleep);
    sleep(time::Duration::from_millis(rand_sleep));

    if (OPT_CPU() != 0) && (OPT_CPU()<=HASH_RNNING_NUM.load(Ordering::SeqCst)+ BELL_RNNING_NUM.load(Ordering::SeqCst) && (!cpu_running()))  {
        debug!("OPT_CPU is {:?}, HASH_RNNING_NUM is {:?}, BELL_RNNING_NUM is {:?}, cpu heavy task is running? {:?}, so we decided to put bellman to cpu",
              OPT_CPU(),HASH_RNNING_NUM.load(Ordering::SeqCst), BELL_RNNING_NUM.load(Ordering::SeqCst), cpu_running());

        true
    } else {
        debug!("OPT_CPU is {:?}, HASH_RNNING_NUM is {:?}, BELL_RNNING_NUM is {:?}, cpu heavy task is running? {:?}, so we decided to MOT put bellman to cpu",
              OPT_CPU(),HASH_RNNING_NUM.load(Ordering::SeqCst), BELL_RNNING_NUM.load(Ordering::SeqCst), cpu_running());
        false
    }
}

fn get_gpu_hash() -> bool
{
    if NO_CUSTOM() {
        return false
    }
    else {
        return get_env_swicher("OPTION4".to_string(), true);
    }

}

fn get_gpu_bell() -> bool
{
    if NO_CUSTOM() {
        return false
    }
    else {
        return get_env_swicher("OPTION5".to_string(), true);
    }
}

fn get_list_from_string(value: String) -> Result<Vec<usize>, serde_json::Error> {
    let res: Vec<usize> = serde_json::from_str(&value)?;
    Ok(res)
}

fn get_gpu_list() -> Vec<usize>
{
    let env_key = "OPTION6";
    let res = if let Ok(option_value) = env::var(env_key) {
        if let Ok(gpu_list) = get_list_from_string(option_value) {
            gpu_list
        } else {
            vec![0, 1, 2, 3]
        }
    } else {
        vec![0, 1, 2, 3]
    };
    res
}

fn get_max_bell_gpu_thread_num() -> usize
{
    let env_key = "OPTION24";
    let def = NUM_PROVING_THREAD()/3;
    let res = if let Ok(num) = env::var(env_key) {
        if let Ok(num) = num.parse() {
            num
        } else {
            def
        }
    } else {
        def
    };
    res
}


fn get_max_verify_thread_num() -> usize
{
    let env_key = "OPTION26";
    let def = 4;
    let res = if let Ok(num) = env::var(env_key) {
        if let Ok(num) = num.parse() {
            num
        } else {
            def
        }
    } else {
        def
    };
    res
}

pub fn NUM_PROVING_THREAD() -> usize{
    get_numeric_env("OPTION3".to_string(), 5)
}
pub fn GPU_HASH() -> bool {
    get_gpu_hash()
}
pub fn GPU_BELL() -> bool {
    get_gpu_bell()
}
pub fn GPU_LIST() -> Vec<usize> {
    get_gpu_list()
}
pub fn GPU_MEM() -> usize {
    get_numeric_env("OPTION7".to_string(), 10500)
}
pub fn PAR_BELL() -> bool {
    false
}
pub fn HASH_FIRST() -> usize {
    get_numeric_env("OPTION9".to_string(), 1)
}
pub fn OPT_CPU() -> usize {
    get_numeric_env("OPTION10".to_string(), 0)
}
pub fn NO_CUSTOM() -> bool {
    get_env_swicher("OPTION12".to_string(), false)
}
pub fn MIN_CPU_NUM() -> usize{
    get_numeric_env("OPTION13".to_string(), 10)
}
pub fn MAX_BELL_GPU_THREAD_NUM() -> usize{
    get_max_bell_gpu_thread_num()
}
pub fn VERIFY_THREAD_NUM() -> usize{
    get_max_verify_thread_num()
}

pub fn SYNTHESIZE_SLEEP_TIME() -> usize {
    get_numeric_env("OPTION_SYNTH_RAND_SLEEP".to_string(), 60000)
}