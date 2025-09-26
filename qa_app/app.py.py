from flask import Flask, render_template, request
from qa_pipeline import rag_qa

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer, context = None, None
    if request.method == "POST":
        question = request.form["question"]
        answer, context = rag_qa(question)
    return render_template("index.html", answer=answer, context=context)

if __name__ == "__main__":
    app.run(debug=True)
