// JavaScript for Sign Language Translator Frontend

let cameraActive = false;
let currentPrediction = '';
let sentence = '';

$(document).ready(function() {
    // Initialize UI state
    updateButtonStates();
    
    // Start Camera Button
    $('#startBtn').click(function() {
        startCamera();
    });
    
    // Stop Camera Button
    $('#stopBtn').click(function() {
        stopCamera();
    });
    
    // Add Word Button
    $('#addWordBtn').click(function() {
        addWordToSentence();
    });
    
    // Clear Button
    $('#clearBtn').click(function() {
        clearSentence();
    });
    
    // Update prediction display periodically
    setInterval(updatePrediction, 500);
});

function startCamera() {
    $.ajax({
        url: '/start_camera',
        type: 'POST',
        success: function(response) {
            cameraActive = true;
            updateButtonStates();
            $('#videoPlaceholder').hide();
            $('#videoFeed').show().attr('src', '/video_feed?' + new Date().getTime());
            showNotification('Camera started successfully!', 'success');
        },
        error: function() {
            showNotification('Failed to start camera', 'error');
        }
    });
}

function stopCamera() {
    $.ajax({
        url: '/stop_camera',
        type: 'POST',
        success: function(response) {
            cameraActive = false;
            updateButtonStates();
            $('#videoFeed').hide().attr('src', '');
            $('#videoPlaceholder').show();
            showNotification('Camera stopped', 'info');
        },
        error: function() {
            showNotification('Failed to stop camera', 'error');
        }
    });
}

function updatePrediction() {
    $.ajax({
        url: '/get_prediction',
        type: 'GET',
        success: function(data) {
            if (data.prediction && data.prediction !== 'None') {
                currentPrediction = data.prediction;
                $('#predictionText').text(data.prediction);
                $('#confidenceText').text('Confidence: ' + (data.confidence * 100).toFixed(1) + '%');
                $('#sentenceInput').val(data.prediction);
                $('#addWordBtn').prop('disabled', false);
                
                // Update confidence color based on value
                if (data.confidence > 0.7) {
                    $('#predictionText').removeClass('text-warning text-danger').addClass('text-success');
                } else if (data.confidence > 0.4) {
                    $('#predictionText').removeClass('text-success text-danger').addClass('text-warning');
                } else {
                    $('#predictionText').removeClass('text-success text-warning').addClass('text-danger');
                }
            } else {
                $('#predictionText').text('Waiting...');
                $('#confidenceText').text('Confidence: 0%');
                $('#predictionText').removeClass('text-success text-warning text-danger').addClass('text-primary');
                $('#addWordBtn').prop('disabled', true);
            }
        },
        error: function() {
            console.error('Failed to get prediction');
        }
    });
}


function addWordToSentence() {
    if (currentPrediction && currentPrediction !== 'None') {
        if (sentence === '') {
            sentence = currentPrediction;
        } else {
            sentence += ' ' + currentPrediction;
        }
        $('#sentenceText').text(sentence);
        $('#sentenceInput').val('');
        showNotification('Word added to sentence', 'success');
    }
}

function clearSentence() {
    sentence = '';
    $('#sentenceText').text('-');
    $('#sentenceInput').val('');
    showNotification('Sentence cleared', 'info');
}

function updateButtonStates() {
    if (cameraActive) {
        $('#startBtn').prop('disabled', true);
        $('#stopBtn').prop('disabled', false);
    } else {
        $('#startBtn').prop('disabled', false);
        $('#stopBtn').prop('disabled', true);
    }
}

function showNotification(message, type) {
    // Create a simple notification
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    const notification = $('<div>')
        .addClass('alert ' + alertClass + ' alert-dismissible fade show position-fixed')
        .css({
            'top': '20px',
            'right': '20px',
            'z-index': '9999',
            'min-width': '300px'
        })
        .html('<strong>' + message + '</strong><button type="button" class="btn-close" data-bs-dismiss="alert"></button>');
    
    $('body').append(notification);
    
    // Auto-dismiss after 3 seconds
    setTimeout(function() {
        notification.alert('close');
    }, 3000);
}

