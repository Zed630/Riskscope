import json
import random
import logging
from typing import List
from swift.rewards import ORM, orms
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[SafeReward] %(message)s',
    handlers=[
        logging.FileHandler("safe_reward_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafeReward(ORM):
    """
    调试专用：返回随机 reward
    目的：确认 Swift 是否正确调用 reward function，并传递梯度
    """
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        batch_size = len(completions)
        
        rewards = []
        for i in range(batch_size):
            try:
                if "<think>\n" not in completions[i]:
                    completion = json.loads(completions[i])
                else:
                    completion = json.loads(completions[i].split("<think>\n")[1])
                # label, catagory = solution[i].split(",")
                _match = re.match(r'([^,]+),\[(.*)\]', solution[i])
                label = _match.group(1)
                catagorys = [_match.group(2).split(', ')]
                label_isharm = label=="Harmful"
                if label_isharm == completion["is_harmful"]:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                    
            except:
                rewards.append(0.0)  # 无法解析的输出，给予最低奖励


        logger.info(f"Safe rewards generated: {rewards}")
        logger.info(f"completions: {[(c[:50] + '...') if len(c) > 50 else c for c in completions]}")
        logger.info(f"solution: {[(s[:50] + '...') if len(s) > 50 else s for s in solution]}")
        return rewards

# 注册到 swift
orms['safe_reward'] = SafeReward