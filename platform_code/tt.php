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
    if(isset($_POST['startDate'],$_POST['endDate'])){
      $startDate=$_POST['startDate'];
      $endDate=$_POST['endDate'];
      $action=$_POST['action'];
      if($action=="none"){
        $sql="select * from detect_log where logDate between '$startDate' and '$endDate' order by logDate desc";
        $result=mysqli_query($mysqli,$sql);
        $data=array();
        if(mysqli_num_rows($result)>0){
          while($row=mysqli_fetch_assoc($result)){
            array_push($data,$row);
          }
          print json_encode($data);
        }
        else{
          echo "No Data";
        }
      }
      else{
        $sql="select * from detect_log where logDate between '$startDate' and '$endDate' and detectAction='$action' order by logDate desc";
        $result=mysqli_query($mysqli,$sql);
        $data=array();
        if(mysqli_num_rows($result)>0){
          while($row=mysqli_fetch_assoc($result)){
            array_push($data,$row);
          }
          print json_encode($data);
        }
        else {
          echo "No Data";
        }
      }
    }
    mysqli_close($mysqli);
?>
