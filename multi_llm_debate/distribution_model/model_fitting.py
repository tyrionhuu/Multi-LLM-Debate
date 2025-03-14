#!/usr/bin/env python
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
