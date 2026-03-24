#!/usr/bin/env python3
# exp14: gamma 0.985 (exp08=0.99에서 소폭 감소)
# exp02 환경 동일, gamma만 변경

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import json, numpy as np, logging
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env_v51_exp02 import ETHTradingEnvV51Exp02
from rl.env_v5 import ETHTradingEnvV5

EXP_NAME = "exp18"; INTERVAL = "30m"; LEVERAGE = 3; TRAIN_END = "2025-09-01"
N_ENVS = 4; TOTAL_STEPS = 3_000_000; GAMMA = 0.975
EXP_DIR = Path(__file__).parent.parent
MODEL_DIR = EXP_DIR / "models" / EXP_NAME; OUT_DIR = EXP_DIR / "experiments"

logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(OUT_DIR / f"{EXP_NAME}.log", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

class CB(BaseCallback):
    def __init__(self, total_steps, check_freq=10000):
        super().__init__(); self.total_steps=total_steps; self.check_freq=check_freq
        self.start_time=None; self.best_reward=-np.inf
    def _on_training_start(self):
        self.start_time=datetime.now()
        log.info(f"[{self.start_time.strftime('%H:%M:%S')}] {EXP_NAME} 학습 시작 (gamma={GAMMA})")
    def _on_step(self):
        progress=self.num_timesteps/self.total_steps
        if progress<0.25: br,ph=0.5,"1-하락(50%)"
        elif progress<0.65: br,ph=0.3,"2-혼합(30%)"
        else: br,ph=0.0,"3-자연"
        self.training_env.env_method("set_bear_ratio",br)
        ne=max(0.01-0.007*progress,0.003); self.model.ent_coef=ne
        if self.num_timesteps%self.check_freq<N_ENVS:
            el=(datetime.now()-self.start_time).total_seconds()
            eta=int(el/max(self.num_timesteps,1)*(self.total_steps-self.num_timesteps)/60)
            mr="N/A"; ne2=len(self.model.ep_info_buffer)
            if ne2>0:
                rr=[ep["r"] for ep in self.model.ep_info_buffer]; mr=f"{np.mean(rr):+.3f}"
                if np.mean(rr)>self.best_reward: self.best_reward=np.mean(rr)
            log.info(f"[{datetime.now().strftime('%H:%M:%S')}] {self.num_timesteps:>9,}/{self.total_steps:,} "
                     f"({progress*100:5.1f}%) | 보상: {mr} | {ph} | ETA: ~{eta}분")
        return True
    def _on_training_end(self):
        el=(datetime.now()-self.start_time).total_seconds()
        log.info(f"\n학습 완료: {el/60:.1f}분 | 최고 보상: {self.best_reward:+.3f}")

def train():
    df=pd.read_csv(PROJECT_ROOT/"rl"/"eth_30m_v41.csv",parse_dates=["time"])
    cutoff=pd.Timestamp(TRAIN_END); si=df[df["time"]<=cutoff].index[-1]+1
    train_df=df.iloc[:si].reset_index(drop=True); test_df=df.iloc[si:].reset_index(drop=True)
    log.info(f"학습: {len(train_df):,}캔들 | 테스트: {len(test_df):,}캔들")
    def make_env(seed):
        def _init():
            env=ETHTradingEnvV51Exp02(train_df,initial_balance=10000.0,leverage=LEVERAGE,
                fee_rate=0.0004,window_size=20,min_hold_steps=4,max_episode_len=2000,
                max_drawdown=0.15,cooldown_steps=8,curriculum=True)
            env.reset(seed=seed); return Monitor(env)
        return _init
    env=DummyVecEnv([make_env(seed=i*42+200) for i in range(N_ENVS)])
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    mp=MODEL_DIR/f"ppo_eth_{INTERVAL}"
    model=PPO("MlpPolicy",env,learning_rate=lambda p:3e-4*(0.3+0.7*p),
        n_steps=2048,batch_size=256,n_epochs=10,gamma=GAMMA,gae_lambda=0.95,
        clip_range=0.2,ent_coef=0.01,vf_coef=0.5,max_grad_norm=0.5,verbose=0,device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[256,128],vf=[256,128])))
    model.learn(total_timesteps=TOTAL_STEPS,callback=CB(TOTAL_STEPS),progress_bar=False)
    model.save(str(mp))
    with open(MODEL_DIR/"meta.json","w") as f:
        json.dump({"experiment":EXP_NAME,"gamma":GAMMA},f,indent=2)
    log.info(f"\n모델 저장: {mp}.zip"); return model,test_df

def backtest(test_df):
    log.info("\n"+"="*60); log.info(f"  백테스트: {EXP_NAME}(g={GAMMA}) vs exp08(g=0.99) vs v5"); log.info("="*60)
    bh=(test_df["close"].iloc[-1]-test_df["close"].iloc[20])/test_df["close"].iloc[20]*100
    results={}
    v5p=PROJECT_ROOT/"rl"/"models"/"v5"/"ppo_eth_30m.zip"
    if v5p.exists():
        ev5=ETHTradingEnvV5(test_df,initial_balance=10000.0,leverage=LEVERAGE,window_size=20,
            min_hold_steps=4,max_episode_len=len(test_df)+100,max_drawdown=1.0,curriculum=False)
        results["v5"]=_bt(ev5,v5p)
    e8p=EXP_DIR/"models"/"exp08"/f"ppo_eth_{INTERVAL}.zip"
    if e8p.exists():
        e8=ETHTradingEnvV51Exp02(test_df,initial_balance=10000.0,leverage=LEVERAGE,window_size=20,
            min_hold_steps=4,max_episode_len=len(test_df)+100,max_drawdown=1.0,cooldown_steps=8,curriculum=False)
        results["exp08"]=_bt(e8,e8p)
    ep=MODEL_DIR/f"ppo_eth_{INTERVAL}.zip"
    if ep.exists():
        ee=ETHTradingEnvV51Exp02(test_df,initial_balance=10000.0,leverage=LEVERAGE,window_size=20,
            min_hold_steps=4,max_episode_len=len(test_df)+100,max_drawdown=1.0,cooldown_steps=8,curriculum=False)
        results[EXP_NAME]=_bt(ee,ep)
    for n,r in results.items():
        s=r["stats"]; d=Counter(r["actions"]); t=len(r["actions"])
        nm={0:"관망",1:"롱",2:"숏",3:"청산"}
        log.info(f"\n{'='*52}\n  {n}\n{'='*52}")
        log.info(f"  수익률: {s['return']:+.2f}% | MDD: {s['mdd']:.2f}% | 거래: {s['trades']}회 | 승률: {s['win_rate']:.1%}")
        log.info(f"  행동: "+" / ".join(f"{nm[k]}:{v}({v/t:.0%})" for k,v in sorted(d.items())))
    if results: _plot(test_df,results,bh)
    return results

def _bt(env,mp):
    model=PPO.load(str(mp),device="cpu"); obs,_=env.reset(); bal=[env.initial_balance]; act=[]; done=False
    while not done:
        a,_=model.predict(obs,deterministic=True); obs,_,te,tr,info=env.step(int(a))
        done=te or tr; bal.append(info["balance"]); act.append(int(a))
    arr=np.array(bal); pk=np.maximum.accumulate(arr); mdd=((arr-pk)/pk*100).min()
    return {"balances":bal,"actions":act,"stats":{"final":bal[-1],"return":(bal[-1]-10000)/10000*100,
            "mdd":mdd,"trades":info["total_trades"],"win_rate":info["win_rate"]}}

def _plot(test_df,results,bh):
    plt.rcParams["font.family"]="DejaVu Sans"
    fig,ax=plt.subplots(1,1,figsize=(14,6))
    fig.suptitle(f"{EXP_NAME} (gamma={GAMMA}) vs exp08 vs v5",fontsize=13,fontweight="bold")
    colors={"v5":"#1565C0","exp08":"#E65100",EXP_NAME:"#2E7D32"}
    for n,r in results.items():
        s=r["stats"]; ax.plot(r["balances"],color=colors.get(n,"#666"),lw=1.8,
            label=f"{n}  {s['return']:+.1f}% (MDD {s['mdd']:.1f}%)")
    ax.axhline(10000,color="#e53935",lw=0.8,ls=":"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); out=OUT_DIR/f"{EXP_NAME}_result.png"
    plt.savefig(out,dpi=150,bbox_inches="tight"); log.info(f"\n그래프: {out}"); plt.close()

if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--backtest-only",action="store_true"); a=p.parse_args()
    if a.backtest_only:
        df=pd.read_csv(PROJECT_ROOT/"rl"/"eth_30m_v41.csv",parse_dates=["time"])
        backtest(df[df["time"]>pd.Timestamp(TRAIN_END)].reset_index(drop=True))
    else: m,td=train(); backtest(td)
