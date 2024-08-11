// static/js/modal.js
window.onload = function() {
    var modal = document.getElementById("modal");
    var span = document.getElementsByClassName("close")[0];
    var closeBtn = document.getElementsByClassName("close-modal-btn")[0];

    if (modal) {
        modal.style.display = "block";

        span.onclick = function() {
            modal.style.display = "none";
        }

        closeBtn.onclick = function() {
            modal.style.display = "none";
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    }
}
