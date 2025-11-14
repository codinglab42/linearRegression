import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def setup_project():
    """Setup del path del progetto - da importare in tutti gli script"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)  # Torna alla root
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"🚀 Project root added to path: {project_root}")
    return project_root

# Esegui automaticamente
project_root = setup_project()


def main():

    print(f"🚀 Inizio del main cost function of house price prediction")




if __name__ == "__main__":
    main()







