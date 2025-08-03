<p align="center">
  <a href="[https://doi.org/your-paper](https://www.biorxiv.org/content/10.1101/2025.01.27.635039v2)"><img src="https://img.shields.io/badge/PAPER-green?style=for-the-badge" alt="arxiv"></a>
  <a href="[https://github.com/yourrepo](https://github.com/AMIRMOHAMMAD-OSS/Phaseek)"><img src="https://img.shields.io/badge/GITHUB-000000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="[https://huggingface.co/yourmodel](https://huggingface.co/AmirMMH/Phaseek)"><img src="https://img.shields.io/badge/HUGGINGFACE-gray?style=for-the-badge&logo=huggingface" alt="HuggingFace"></a>
  <a href="[https://colab.research.google.com/your-notebook](https://colab.research.google.com/gist/AMIRMOHAMMAD-OSS/22f560f374f08786c6c5992887cb7ca7/phaseek.ipynb)"><img src="https://img.shields.io/badge/COLAB-red?style=for-the-badge&logo=googlecolab" alt="Colab"></a>
</p>


**Follow these steps to test:**

1. Install python `virtualenv` package globally if you dont have it installed (in linux distros probably you should install global packages with your package manager not pip)

2. Clone the project -> `git clone <project-clone-url> ~/Phaseek
`

3. Go to cloned folder; i.e -> `cd Phaseek`

4. Create a virtual environment there `virtualenv .venv` or `python -m virtualenv` if did not work

5. Activate the virtual environment `source .venv/bin/activate` (Linux & Mac)

6. Instal the required packages `pip install -r requirements.txt`

7. Run the `Function/runner.py`

   
**Example:**

```shell
python Function/runner.py --sequence "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGYGSGRGKGGKGLGGKGLGKGGAKRHRK" --id test_sequence --directory test_results

```
