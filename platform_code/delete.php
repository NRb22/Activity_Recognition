<?php
  header('Content-Type: text/html; charset=utf-8');
?>
<?php session_start();
    DEFINE('DB_USERNAME', 'root');
    DEFINE('DB_PASSWORD', '');
    DEFINE('DB_HOST', 'localhost');
    DEFINE('DB_DATABASE', 'LearningEyes');
    $mysqli = new mysqli(DB_HOST, DB_USERNAME, DB_PASSWORD, DB_DATABASE);
    if (mysqli_connect_error()) {
        die('Connect Error ('.mysqli_connect_errno().') '.mysqli_connect_error());
    }
    $no=$_POST['no'];
    $userpw=$_POST['pass'];
    //$pass=$_POST['']
    $sql="delete from board where id='$no' and pass='$userpw'";
    $result=mysqli_query($mysqli,$sql);
    mysqli_close($mysqli);
    ?>
     <meta http-equiv='Refresh' content='0; URL=main.php'>
