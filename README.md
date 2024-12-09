#  Install package
```
pip install -r requirements.txt
```

# Make sure placing stl files
You should locate stl files under low_cost_robot/assets/
I didn't add *.stl files under that directory because sti file has
relatively huge file size.
```
low_cost_robot/
├─ assets/
│  ├─ *.stl
    ...
```

# Run mujoco simulation in mac OS
- use `mjpython` binary file instead of `python3`
- `mjpython` is located under the package `mujoco`

# interface branch
## requirements
```bash
pip install openai
```
