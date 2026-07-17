// The browser reset action clears the four form fields. Server-side validation remains in Flask.
document.getElementById("prediction-form").addEventListener("reset", function () {
    window.setTimeout(function () {
        document.querySelectorAll(".message").forEach(function (message) {
            message.remove();
        });
    }, 0);
});
