$(document).ready(function () {
    // Hide elements initially
    $('.loader, #result, .predict-button-container, #imagePreview').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result).show();
                $('#uploadPlaceholder').hide();
                $('.predict-button-container').show(); // Show predict button container after image upload
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('#result').hide().text('');
        readURL(this);
    });

    // Predict button click
    $('#btn-predict').click(function () {
        const form_data = new FormData($('#upload-file')[0]);
        $(this).hide();
        $('.loader').fadeIn();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();

                // Prepare the result modal
                let resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
                let modalDialog = document.querySelector('.modal-content');
                let resultText = document.getElementById('resultText');

                // Set result text and modal color based on prediction
                if (data.includes("Yes")) {
                    resultText.innerHTML = 'Brain Tumor Detected!';
                    modalDialog.style.backgroundColor = 'red';
                    modalDialog.style.color = 'white';
                } else {
                    resultText.innerHTML = 'Brain Tumor Not Detected!';
                    modalDialog.style.backgroundColor = 'green';
                    modalDialog.style.color = 'white';
                }

                // Show the modal
                resultModal.show();

                console.log('Success!');
            },
            error: function () {
                $('.loader').hide();
                alert('An error occurred. Please try again.');
            }
        });
    });
});
