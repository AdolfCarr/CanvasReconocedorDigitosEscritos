$("#paint").css({ width: "300px" });
//$("#number").css({ width: "150px", "font-size": "120px" });
//$("#clear").css({ "font-size": "20px" });

var cw = $("#paint").width();
$("#paint").css({ height: cw + "px" });

//cw = $("#number").width();
//$("#number").css({ height: cw + "px" });

var canvas = document.getElementById("myCanvas");
var context = canvas.getContext("2d");

var compuetedStyle = getComputedStyle(document.getElementById("paint"));
canvas.width = parseInt(compuetedStyle.getPropertyValue("width"));
canvas.height = parseInt(compuetedStyle.getPropertyValue("height"));

var mouse = { x: 0, y: 0 };

canvas.addEventListener(
  "mousemove",
  function (e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
  },
  false
);

context.lineWidth = 10;
context.lineJoin = "round";
context.lineCap = "round";
context.strokeStyle = "#b8e994";

canvas.addEventListener(
  "mousedown",
  function (e) {
    context.moveTo(mouse.x, mouse.y);
    context.beginPath();
    canvas.addEventListener("mousemove", onPaint, false);
  },
  false
);

canvas.addEventListener(
  "mouseup",
  function () {
    canvas.removeEventListener("mousemove", onPaint, false);
    var img = new Image();
    img.onload = function () {
      context.drawImage(img, 0, 0, 28, 28);
      data = context.getImageData(0, 0, 28, 28).data;
      var input = [];
      for (var i = 0; i < data.length; i += 4) {
        input.push(data[i + 2] / 255);
      }
      predict(input);
      predictConv20(input);
      predictConv60(input);
      predictConv200(input);
    };
    img.src = canvas.toDataURL("image/png");
  },
  false
);

var onPaint = function () {
  context.lineTo(mouse.x, mouse.y);
  context.stroke();
};

tf.loadLayersModel("model_sequencial/model.json").then(function (model) {
  window.model = model;
});

tf.loadLayersModel("modelo_convolucional20/model.json").then(function (model) {
  window.modelConv20 = model;
});

tf.loadLayersModel("model_convolucional/model.json").then(function (model) {
  window.modelConv60 = model;
});

tf.loadLayersModel("model_convolucional200/model.json").then(function (model) {
  window.modelConv200 = model;
});



var predict = function (input) {
  if (window.model) {
    window.model
      .predict([tf.tensor(input).reshape([1, 28, 28, 1])])
      .array()
      .then(function (scores) {
        scores = scores[0];
        predicted = scores.indexOf(Math.max(...scores));
        $("#sequential_prediction").html(predicted);
      });
  } else {
    // The model takes a bit to load, if we are too fast, wait
    setTimeout(function () {
      predict(input);
    }, 50);
  }
};

var predictConv20 = function (input) {
  if (window.modelConv20) {
    window.modelConv20
      .predict([tf.tensor(input).reshape([1, 28, 28, 1])])
      .array()
      .then(function (scores) {
        scores = scores[0];
        predicted = scores.indexOf(Math.max(...scores));
        $("#convolutional_prediction20").html(predicted);
      });
  } else {
    // The model takes a bit to load, if we are too fast, wait
    setTimeout(function () {
      predictConv20(input);
    }, 50);
  }
};

var predictConv60 = function (input) {
  if (window.modelConv60) {
    window.modelConv60
      .predict([tf.tensor(input).reshape([1, 28, 28, 1])])
      .array()
      .then(function (scores) {
        scores = scores[0];
        predicted = scores.indexOf(Math.max(...scores));
        $("#convolutional_prediction60").html(predicted);
      });
  } else {
    // The model takes a bit to load, if we are too fast, wait
    setTimeout(function () {
      predictConv60(input);
    }, 50);
  }
};

var predictConv200 = function (input) {
  if (window.modelConv200) {
    window.modelConv200
      .predict([tf.tensor(input).reshape([1, 28, 28, 1])])
      .array()
      .then(function (scores) {
        scores = scores[0];
        predicted = scores.indexOf(Math.max(...scores));
        $("#convolutional_prediction200").html(predicted);
      });
  } else {
    // The model takes a bit to load, if we are too fast, wait
    setTimeout(function () {
      predictConv200(input);
    }, 50);
  }
};

$("#clear").click(function () {
  context.clearRect(0, 0, canvas.width, canvas.height);
  $("#sequential_prediction").html("");
  $("#convolutional_prediction20").html("");
  $("#convolutional_prediction60").html("");
  $("#convolutional_prediction200").html("");
});

