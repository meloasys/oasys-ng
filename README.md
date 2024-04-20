# oasys-ng
- RL engine for Oasys
- testing with DI-engine (https://github.com/opendilab/DI-engine)


contents
```
|-- README.md
|-- contents
|   |-- configs
|   |-- env
|   `-- train_dqn.py
|-- data
|   `-- train_data.csv
|-- logs
|   `-- logs
`-- src
```

Mod list
1. create single logger during process
    - problem
        - while procs on each different experiment with diff configuration, logger writes same experment name. to fix this follow below.
    - fix
        ```python
        # add cfg on online_logger function
        task.use(online_logger(train_show_freq=10, cfg=cfg))

        # DI-engine/ding/framework/middleware/functional/logger.py
        # 22
        def online_logger(record_train_iter: bool = False, train_show_freq: int = 100, *args, **kwrgs) -> Callable:
        # 40
        _cfg = kwrgs['cfg']
        writer = DistributedWriter.get_instance(_cfg.exp_name)
        # do same on offline_logger function
        ```