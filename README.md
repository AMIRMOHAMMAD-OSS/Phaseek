**Follow these steps to test:**

1. Install python `virtualenv` package globally if you dont have it installed (in linux distros probably you should install global packages with your package manager not pip)

2. Clone the project -> `git clone <project-clone-url> ~/Phaseek
`

3. Go to cloned folder; i.e -> `cd Phaseek`

4. Create a virtual environment there `virtualenv .venv` or `python -m virtualenv` if did not work

5. Activate the virtual environment `source .venv/bin/activate` (Linux & Mac)

6. Run `runner.py` with desired values

**Example:**

```shell
python runner.py --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGYGSGRGKGGKGLGGKGLGKGGAKRHRK" --id test_sequence --directory test_results

```
